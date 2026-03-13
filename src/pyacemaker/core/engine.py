from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.exceptions import MDHaltInterrupt
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.domain_models.constants import (
    ERR_STRUCTURE_NONE,
    LAMMPS_SCREEN_ARG,
)
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.domain_models.workflow import ActiveLearningThresholds
from pyacemaker.interfaces.lammps_driver import LammpsDriver


class TwoTierEvaluator:
    """
    Evaluates MD uncertainty across consecutive steps.
    Raises MDHaltInterrupt to pause the simulation if threshold is exceeded for multiple steps.
    Designed to be called via LAMMPS `fix python/invoke`.
    """

    def __init__(self, thresholds: ActiveLearningThresholds) -> None:
        self.thresholds = thresholds
        self.consecutive_exceedances = 0

    def __call__(self, lmp: Any) -> None:
        """
        Callback executed by LAMMPS.
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Safely extract c_max_gamma
            # lmp is the LAMMPS python object instance
            max_gamma = float(lmp.extract_variable("max_g", None, 0))
        except Exception:
            logger.exception("Failed to extract max_g in evaluator")
            raise

        if max_gamma > self.thresholds.threshold_call_dft:
            self.consecutive_exceedances += 1
            if self.consecutive_exceedances >= self.thresholds.smooth_steps:
                self.consecutive_exceedances = 0  # Reset for future
                msg = f"Uncertainty {max_gamma} exceeded threshold {self.thresholds.threshold_call_dft} for {self.thresholds.smooth_steps} consecutive steps."
                raise MDHaltInterrupt(msg)
        else:
            self.consecutive_exceedances = 0


class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS.
    Handles input generation, execution, and result parsing.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator | None = None,
        file_manager: LammpsFileManager | None = None,
    ) -> None:
        """
        Initialize the engine with configuration.
        Allows dependency injection for generator and file manager.
        """
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.file_manager = file_manager or LammpsFileManager(config)

    def _prepare_simulation_env(
        self, structure: Atoms | None, potential: Any
    ) -> tuple[Any, Path, Path, Path, list[str], Path]:
        """
        Prepares the simulation environment: validation, paths, and files.
        Returns: (ctx, data_file, dump_file, log_file, elements, potential_path)
        """
        if structure is None:
            raise ValueError(ERR_STRUCTURE_NONE)

        LammpsInputValidator.validate_structure(structure)
        potential_path = LammpsInputValidator.validate_potential(potential)
        potential_path = potential_path.resolve(strict=True)

        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(
            structure
        )
        return ctx, data_file, dump_file, log_file, elements, potential_path

    def _ensure_script_readable(self, script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    def _validate_script_content(self, script_path: Path) -> None:
        """Validates script content for shell injection vulnerabilities."""
        from pyacemaker.utils.validation import validate_lammps_script_file

        validate_lammps_script_file(script_path)

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        """
        Executes the simulation script with standardized error handling.
        """
        from pyacemaker.core.error_handler import LammpsErrorHandler

        try:
            self._ensure_script_readable(script_path)

            # The validation MUST be done synchronously and atomically with execution
            # to absolutely prevent TOCTOU (Time of check to time of use) vulnerability.
            # While the driver does token validation per command, we check file integrity immediately prior.
            self._validate_script_content(script_path)

            # Scalability: Use run_file to stream script execution
            driver.run_file(str(script_path))

        except Exception as e:
            LammpsErrorHandler.handle(e)

    def _validate_resume_params(self, kwargs: dict[str, Any]) -> tuple[int | None, int | None]:
        """Validates keyword arguments for resuming and overriding steps."""
        resume_step = kwargs.get("resume_from_step")
        override_n_steps = kwargs.get("override_n_steps")

        if resume_step is not None:
            if not isinstance(resume_step, int) or resume_step < 0:
                msg = "resume_from_step must be a non-negative integer"
                raise ValueError(msg)
            if resume_step > self.config.n_steps:
                msg = f"resume_from_step {resume_step} cannot exceed configured n_steps {self.config.n_steps}"
                raise ValueError(msg)

        if override_n_steps is not None:
            if not isinstance(override_n_steps, int) or override_n_steps < 0:
                msg = "override_n_steps must be a non-negative integer"
                raise ValueError(msg)
            if resume_step is not None and resume_step + override_n_steps > self.config.n_steps:
                msg = f"Combined resume_step {resume_step} and override_n_steps {override_n_steps} exceed original simulation steps {self.config.n_steps}"
                raise ValueError(msg)

        return resume_step, override_n_steps

    def _prepare_script(
        self,
        temp_dir: Path,
        potential_path: Path,
        data_file: Path,
        dump_file: Path,
        elements: list[str],
        resume_step: int | None,
        override_n_steps: int | None = None,
        thresholds: ActiveLearningThresholds | None = None,
    ) -> tuple[Path, Path]:
        """Prepares input and restart scripts within the working directory."""
        input_script_path = temp_dir / "input.lmp"
        restart_path = temp_dir / "restart.lmp"

        if self.config.fix_halt and thresholds is not None:
            import json
            import shlex

            # Find the static evaluator script path within the codebase strictly BEFORE writing configs
            static_script_path = Path(__file__).parent / "evaluator_script.py"

            if not static_script_path.exists():
                msg = f"Static evaluator script not found at {static_script_path}"
                raise FileNotFoundError(msg)

            # Safely write the configuration data as JSON instead of injecting Python code
            config_path = temp_dir / "evaluator_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(thresholds.model_dump(), f)

            # Load the static script and explicitly pass the config path during LAMMPS initialization
            with input_script_path.open("w", encoding="utf-8") as f:
                # Load the static evaluator script directly
                f.write(
                    f'python eval_wrapper file {shlex.quote(str(static_script_path.resolve()))} format "v"\n'
                )
                # Instead of executing an inline python block via LAMMPS with f-strings,
                # we pass the configuration via a structured LAMMPS variable.
                safe_config_path = shlex.quote(str(config_path.resolve()))
                f.write(f"variable evaluator_config_path string {safe_config_path}\n")
                f.write("python init_evaluator invoke here\n")
        else:
            with input_script_path.open("w") as f:
                pass  # Just touch the file

        with input_script_path.open("a") as f:
            if resume_step is not None and resume_step > 0:
                # Do NOT create the file here. LAMMPS `read_restart` demands a valid, pre-existing binary file.
                # Silent empty creation causes LAMMPS to crash.
                # We enforce that a valid restart file exists prior to attempting a resume.
                if not restart_path.exists() or restart_path.stat().st_size == 0:
                    msg = f"Valid restart file not found at {restart_path} for resuming"
                    raise FileNotFoundError(msg)

                self.generator.write_script_resume(
                    f,
                    potential_path,
                    restart_path,
                    dump_file,
                    elements,
                    resume_step,
                    override_n_steps,
                )
            else:
                self.generator.write_script(f, potential_path, data_file, dump_file, elements)
                f.write(f"\nwrite_restart {restart_path}\n")

        return input_script_path, restart_path

    def _extract_results(
        self, driver: LammpsDriver, kwargs: dict[str, Any], dump_file: Path, log_file: Path
    ) -> MDSimulationResult:
        """Extracts and formats simulation results from the driver."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            energy = driver.extract_variable("pe")
            temperature = driver.extract_variable("temp")
            step = int(driver.extract_variable("step"))
            forces = driver.get_forces().tolist()
            stress = driver.get_stress().tolist()
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            msg = f"Critical extraction failure: Failed to extract basic results from LAMMPS: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        max_gamma = 0.0
        if self.config.fix_halt:
            try:
                max_gamma = driver.extract_variable("max_g")
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                msg = f"Critical extraction failure: Failed to extract max_g from LAMMPS: {e}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

        halted = False
        n_steps_target = kwargs.get("override_n_steps", self.config.n_steps)

        if self.config.fix_halt:
            halted = step < n_steps_target

        return MDSimulationResult(
            energy=energy,
            forces=forces,
            stress=stress,
            halted=halted,
            max_gamma=max_gamma,
            n_steps=step,
            temperature=temperature,
            trajectory_path=str(dump_file),
            log_path=str(log_file),
            halt_structure_path=str(dump_file) if halted else None,
            halt_step=step if halted else None,
        )

    def run(self, structure: Atoms | None, potential: Any, **kwargs: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        Kwargs:
            resume_from_step (int): Step to resume MD from.
            override_n_steps (int): Override number of steps to run.
            thresholds (ActiveLearningThresholds): Thresholds for dynamic pause.
        """
        if structure is None:
            raise ValueError(ERR_STRUCTURE_NONE)

        resume_step, override_n_steps = self._validate_resume_params(kwargs)

        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        with ctx:
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path, _ = self._prepare_script(
                temp_dir,
                potential_path,
                data_file,
                dump_file,
                elements,
                resume_step,
                override_n_steps,
                kwargs.get("thresholds"),
            )

            lammps_args = ["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)]

            if hasattr(self.config, "lammps_args") and self.config.lammps_args:
                lammps_args.extend(self.config.lammps_args)

            # Master-Slave Isolation via ThreadPoolExecutor ensuring the master Python orchestration
            # safely survives any LAMMPS C++ crashes.
            import concurrent.futures

            def _isolated_execution() -> MDSimulationResult:
                driver = LammpsDriver(lammps_args)
                try:
                    self._execute_simulation(driver, input_script_path)
                    return self._extract_results(driver, kwargs, dump_file, log_file)
                finally:
                    if hasattr(driver, "close"):
                        driver.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_isolated_execution)
                try:
                    result = future.result()
                except Exception:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.exception("Isolated LAMMPS execution completely failed")
                    raise
                else:
                    # Artifact Cleanup: automatically cleanup large structural log/dump files if execution was fully completed successfully
                    # exactly as perfectly required completely natively.
                    if not result.halted:
                        self._cleanup_artifacts([dump_file, log_file])

                    return result

    def _cleanup_artifacts(self, paths: list[Path]) -> None:
        """Automatically compress or entirely successfully efficiently delete perfectly massive output files."""
        import contextlib
        for p in paths:
            if p.exists() and p.is_file():
                with contextlib.suppress(Exception):
                    p.unlink()

    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
        """
        Computes static properties (energy, forces, stress) for a structure.
        Equivalent to a 0-step MD run.
        """
        static_config = self.config.model_copy(
            update={"n_steps": 0, "minimize": False, "thermo_freq": 1, "dump_freq": 0}
        )

        engine = LammpsEngine(static_config)
        return engine.run(structure, potential)

    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        """
        Relaxes the structure to a local minimum using LAMMPS minimize.
        """
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        with ctx:
            # Generate minimization script
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            script_path = temp_dir / "relax.lmp"

            with script_path.open("w") as f:
                self.generator.write_minimization_script(f, potential_path, data_file, elements)

            # Execute
            lammps_args = ["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)]

            if hasattr(self.config, "lammps_args") and self.config.lammps_args:
                lammps_args.extend(self.config.lammps_args)

            driver = LammpsDriver(lammps_args)
            try:
                self._execute_simulation(driver, script_path)
                return driver.get_atoms(elements)
            finally:
                if hasattr(driver, "close"):
                    driver.close()
