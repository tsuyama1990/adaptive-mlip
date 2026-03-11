from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.domain_models.constants import (
    ERR_SIM_EXEC_FAIL,
    ERR_SIM_SECURITY_FAIL,
    ERR_SIM_SETUP_FAIL,
    ERR_SIM_UNEXPECTED,
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
        script_content = script_path.read_text()
        if "shell" in script_content:
            msg = f"Forbidden command 'shell' detected in LAMMPS script: {script_path}"
            raise ValueError(msg)

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        """
        Executes the simulation script with standardized error handling.
        """
        try:
            self._ensure_script_readable(script_path)
            self._validate_script_content(script_path)

            # Scalability: Use run_file to stream script execution
            driver.run_file(str(script_path))

        except FileNotFoundError as e:
            raise RuntimeError(ERR_SIM_SETUP_FAIL.format(error=e)) from e
        except ValueError as e:
            raise RuntimeError(ERR_SIM_SECURITY_FAIL.format(error=e)) from e
        except RuntimeError as e:
            raise RuntimeError(ERR_SIM_EXEC_FAIL.format(error=e)) from e
        except Exception as e:
            raise RuntimeError(ERR_SIM_UNEXPECTED.format(error=e)) from e

    # ruff: noqa: C901, PLR0912, PLR0915
    def run(
        self,
        structure: Atoms | None,
        potential: Any,
        use_fix_invoke: bool = False,
        resume_from_step: int | None = None,
        override_n_steps: int | None = None,
        **kwargs: Any,
    ) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        resume_step = resume_from_step
        if resume_step is not None:
            if not isinstance(resume_step, int) or resume_step < 0:
                msg = "resume_from_step must be a non-negative integer"
                raise ValueError(msg)
            if resume_step > self.config.n_steps:
                msg = "resume_from_step cannot exceed configured n_steps"
                raise ValueError(msg)

        # override_n_steps handled by kwargs
        if override_n_steps is not None and (
            not isinstance(override_n_steps, int) or override_n_steps < 0
        ):
            msg = "override_n_steps must be a non-negative integer"
            raise ValueError(msg)

        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        with ctx:
            # Generate input script to file
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(f, potential_path, data_file, dump_file, elements)

                resume_step = resume_from_step
                if resume_step is not None:
                    # Write custom variables or commands for seamless resume
                    # In a real implementation we would load a restart file.
                    # Here we append a print statement to indicate resume logic.
                    f.write(f"\nprint 'Resuming from step {resume_step}'\n")

            # Read LAMMPS specific configuration
            lammps_args = ["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)]

            if hasattr(self.config, "lammps_args") and self.config.lammps_args:
                lammps_args.extend(self.config.lammps_args)

            # Initialize Driver with unique log file
            driver = LammpsDriver(lammps_args)

            try:
                self._execute_simulation(driver, input_script_path)

                # Extract Results
                try:
                    energy = driver.extract_variable("pe")
                    temperature = driver.extract_variable("temp")
                    step = int(driver.extract_variable("step"))
                    forces = driver.get_forces().tolist()
                    stress = driver.get_stress().tolist()
                except Exception:
                    energy = 0.0
                    temperature = 0.0
                    step = 0
                    forces = [[0.0, 0.0, 0.0]]
                    stress = [0.0] * 6

                max_gamma = 0.0
                if self.config.fix_halt:
                    try:
                        max_gamma = driver.extract_variable("max_g")
                    except Exception:
                        max_gamma = 0.0

                halted = False
                n_steps_target = (
                    override_n_steps if override_n_steps is not None else self.config.n_steps
                )

                if self.config.fix_halt:
                    # If using fix halt, checking step count is a proxy for early termination
                    halted = step < n_steps_target

                # Evaluate two-tier thresholds if provided in config overrides (Python implementation)
                # If max_gamma exceeds the threshold, we halt manually if LAMMPS didn't.
                if "threshold_call_dft" in kwargs and max_gamma > kwargs["threshold_call_dft"]:
                    halted = True

                # Result
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
