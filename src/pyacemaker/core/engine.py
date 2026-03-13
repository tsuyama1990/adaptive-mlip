from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.domain_models.constants import (
    ERR_STRUCTURE_NONE,
    LAMMPS_SCREEN_ARG,
)
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.interfaces.lammps_driver import LammpsDriver


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
            self._validate_script_content(script_path)

            # Scalability: Use run_file to stream script execution
            driver.run_file(str(script_path))

        except Exception as e:
            LammpsErrorHandler.handle(e)

    def _validate_resume_params(self, kwargs: dict[str, Any]) -> tuple[int | None, int | None]:
        """Validates keyword arguments for resuming and overriding steps."""
        resume_step = kwargs.get("resume_from_step")
        if resume_step is not None:
            if not isinstance(resume_step, int) or resume_step < 0:
                msg = "resume_from_step must be a non-negative integer"
                raise ValueError(msg)
            if resume_step > self.config.n_steps:
                msg = "resume_from_step cannot exceed configured n_steps"
                raise ValueError(msg)

        override_n_steps = kwargs.get("override_n_steps")
        if override_n_steps is not None and (
            not isinstance(override_n_steps, int) or override_n_steps < 0
        ):
            msg = "override_n_steps must be a non-negative integer"
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
    ) -> tuple[Path, Path]:
        """Prepares input and restart scripts within the working directory."""
        input_script_path = temp_dir / "input.lmp"
        restart_path = temp_dir / "restart.lmp"

        with input_script_path.open("w") as f:
            if resume_step is not None and resume_step > 0:
                # Do NOT create the file here. LAMMPS `read_restart` demands a valid, pre-existing binary file.
                # Silent empty creation causes LAMMPS to crash.
                # We enforce that a valid restart file exists prior to attempting a resume.
                if not restart_path.exists() or restart_path.stat().st_size == 0:
                    msg = f"Valid restart file not found at {restart_path} for resuming"
                    raise FileNotFoundError(msg)

                self.generator.write_script_resume(
                    f, potential_path, restart_path, dump_file, elements, resume_step
                )
            else:
                self.generator.write_script(f, potential_path, data_file, dump_file, elements)
                f.write(f"\nwrite_restart {restart_path}\n")

        return input_script_path, restart_path

    def _extract_results(
        self, driver: LammpsDriver, kwargs: dict[str, Any], dump_file: Path, log_file: Path
    ) -> MDSimulationResult:
        """Extracts and formats simulation results from the driver."""
        try:
            energy = driver.extract_variable("pe")
            temperature = driver.extract_variable("temp")
            step = int(driver.extract_variable("step"))
            forces = driver.get_forces().tolist()
            stress = driver.get_stress().tolist()
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to extract basic results from LAMMPS: {e}")
            energy = 0.0
            temperature = 0.0
            step = 0
            forces = [[0.0, 0.0, 0.0]]
            stress = [0.0] * 6

        max_gamma = 0.0
        if self.config.fix_halt:
            try:
                max_gamma = driver.extract_variable("max_g")
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to extract max_g from LAMMPS: {e}")
                max_gamma = 0.0

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
                temp_dir, potential_path, data_file, dump_file, elements, resume_step
            )

            lammps_args = ["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)]

            if hasattr(self.config, "lammps_args") and self.config.lammps_args:
                lammps_args.extend(self.config.lammps_args)

            driver = LammpsDriver(lammps_args)

            try:
                self._execute_simulation(driver, input_script_path)
                return self._extract_results(driver, kwargs, dump_file, log_file)
            finally:
                if hasattr(driver, "close"):
                    driver.close()

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
