import logging
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.domain_models.constants import (
    LAMMPS_SCREEN_ARG,
)
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.domain_models.workflow import OTFConfig
from pyacemaker.interfaces.lammps_driver import LammpsDriver
from pyacemaker.utils.io_transaction import DirectoryTransaction


class SimulationExecutionStrategy:
    """Strategy for executing LAMMPS simulations with proper error handling and context logging."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, driver: LammpsDriver | None, script_path: Path) -> None:
        """
        Executes the simulation script using the provided driver.
        Validates inputs and handles errors gracefully without obscuring original traces.
        """
        if driver is None:
            msg = "Driver instance cannot be None."
            self.logger.error(msg)
            raise ValueError(msg)

        if not script_path.exists():
            msg = "Simulation script not found."
            self.logger.error(f"{msg} Path: {script_path}")
            raise FileNotFoundError(msg)

        self.logger.info("Executing LAMMPS simulation", extra={"script_path": str(script_path)})

        # We allow specific exceptions to propagate naturally
        # Scalability: Use run_file to stream script execution
        driver.run_file(str(script_path))


class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS.
    Handles input generation, execution, and result parsing.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator,
        file_manager: LammpsFileManager,
        execution_strategy: SimulationExecutionStrategy | None = None,
        otf_config: OTFConfig | None = None,
    ) -> None:
        """
        Initialize the engine with configuration.
        Requires strict dependency injection for generator and file manager.
        """
        if config is None:
            msg = "MDConfig cannot be None"
            raise ValueError(msg)
        if generator is None:
            msg = "LammpsScriptGenerator cannot be None"
            raise ValueError(msg)
        if not isinstance(generator, LammpsScriptGenerator):
            raise TypeError("generator must be an instance of LammpsScriptGenerator")

        if file_manager is None:
            msg = "LammpsFileManager cannot be None"
            raise ValueError(msg)
        if not isinstance(file_manager, LammpsFileManager):
            raise TypeError("file_manager must be an instance of LammpsFileManager")

        if execution_strategy is not None and not isinstance(execution_strategy, SimulationExecutionStrategy):
            raise TypeError("execution_strategy must be an instance of SimulationExecutionStrategy")

        self.config = MDConfig.model_validate(config)
        self.generator = generator
        self.file_manager = file_manager
        self.execution_strategy = execution_strategy or SimulationExecutionStrategy()
        self.otf_config = otf_config

    def _prepare_simulation_env(
        self, structure: Atoms | None, potential: Any
    ) -> tuple[DirectoryTransaction, Path, Path, Path, list[str], Path]:
        """
        Prepares the simulation environment: validation, paths, and files.
        Returns: (ctx, data_file, dump_file, log_file, elements, potential_path)
        """

        LammpsInputValidator.validate_structure(structure)
        potential_path = LammpsInputValidator.validate_potential(potential)
        potential_path = potential_path.resolve(strict=True)

        # We checked structure is not None inside LammpsInputValidator.validate_structure
        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(
            structure  # type: ignore[arg-type]
        )
        return ctx, data_file, dump_file, log_file, elements, potential_path

    def _ensure_script_readable(self, script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    def save_state(self, path: str | Path) -> None:
        """
        Saves the internal state of the engine.
        """

    def load_state(self, path: str | Path) -> None:
        """
        Loads the internal state of the engine.
        """

    def run(self, structure: Atoms | None, potential: Any, **kwargs: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        If resume_from_step is provided, skips initialization and applies a soft-start.
        """
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        resume_from_step = kwargs.get("resume_from_step")
        use_python_invoke = kwargs.get("use_python_invoke", False)

        with ctx:
            # Generate input script to file
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(f, potential_path, data_file, dump_file, elements)

                # Use fix python/invoke for true Master-Slave inversion if configured
                if use_python_invoke:
                    f.write("\n# Master-Slave Inversion via fix python/invoke\n")
                    f.write("python check_uncertainty file pyacemaker_callback.py\n")
                    f.write("fix check_otf all python/invoke 10 post_force check_uncertainty\n")
                elif resume_from_step is not None:
                    # Fallback for subprocess simulated resume
                    f.write(f"\n# Seamless Resume from step {resume_from_step}\n")
                    f.write("fix soft_start all langevin 300.0 300.0 100.0 48279\n")
                    f.write("run 50\n")
                    f.write("unfix soft_start\n")

            # Initialize Driver with unique log file
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            try:
                self.execution_strategy.execute(driver, input_script_path)

                # Extract Results
                try:
                    energy = driver.extract_variable("pe")
                    temperature = driver.extract_variable("temp")
                    step = int(driver.extract_variable("step"))
                    forces = driver.get_forces().tolist()
                    stress = driver.get_stress().tolist()
                except (ValueError, TypeError, KeyError) as e:
                    import logging

                    msg = "LAMMPS extraction failed: missing or invalid variables."
                    logging.getLogger(__name__).exception(msg)
                    raise RuntimeError(msg) from e

                max_gamma = 0.0
                if self.otf_config and self.otf_config.fix_halt:
                    try:
                        max_gamma = driver.extract_variable("max_g")
                    except (ValueError, TypeError, KeyError):
                        max_gamma = 0.0

                halted = False
                if self.otf_config and self.otf_config.fix_halt:
                    # If using fix halt, checking step count is a proxy for early termination
                    halted = step < self.config.n_steps

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

        generator = LammpsScriptGenerator(static_config)
        file_manager = LammpsFileManager(static_config)
        engine = LammpsEngine(static_config, generator, file_manager)
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
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self.execution_strategy.execute(driver, script_path)
                return driver.get_atoms(elements)
            finally:
                if hasattr(driver, "close"):
                    driver.close()
