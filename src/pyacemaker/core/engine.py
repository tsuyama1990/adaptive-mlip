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


class LammpsExecutor:
    """Handles executing LAMMPS scripts."""

    @staticmethod
    def _ensure_script_readable(script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    @staticmethod
    def execute_simulation(driver: LammpsDriver, script_path: Path) -> None:
        try:
            LammpsExecutor._ensure_script_readable(script_path)
            driver.run_file(str(script_path))
        except FileNotFoundError as e:
            raise RuntimeError(ERR_SIM_SETUP_FAIL.format(error=e)) from e
        except ValueError as e:
            raise RuntimeError(ERR_SIM_SECURITY_FAIL.format(error=e)) from e
        except RuntimeError as e:
            raise RuntimeError(ERR_SIM_EXEC_FAIL.format(error=e)) from e
        except Exception as e:
            raise RuntimeError(ERR_SIM_UNEXPECTED.format(error=e)) from e

class LammpsResultParser:
    """Handles extracting results from LAMMPS driver."""

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def parse_md_result(self, driver: LammpsDriver, dump_file: Path, log_file: Path) -> MDSimulationResult:
        try:
            energy = driver.extract_variable("pe")
            temperature = driver.extract_variable("temp")
            step = int(driver.extract_variable("step"))
            # Scalability fix: convert to simple python lists incrementally to avoid large numpy materializations
            # if driver supports returning an iterator, but driver.get_forces returns numpy array right now.
            # Using driver.get_forces().tolist() directly on huge arrays could use 2x memory,
            # but we need it as a python list for Pydantic. We do it carefully.
            forces_array = driver.get_forces()
            forces = [list(f) for f in forces_array]

            stress_array = driver.get_stress()
            stress = list(stress_array)
        except Exception:
            energy = 0.0
            temperature = 0.0
            step = 0
            forces = self.config.default_forces
            stress = [0.0] * 6

        max_gamma = 0.0
        if self.config.fix_halt:
            try:
                max_gamma = driver.extract_variable("max_g")
            except Exception:
                max_gamma = 0.0

        halted = False
        if self.config.fix_halt:
            halted = step < self.config.n_steps

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


class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS.
    Composes generation, execution, and result parsing.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator | None = None,
        file_manager: LammpsFileManager | None = None,
        executor: LammpsExecutor | None = None,
        parser: LammpsResultParser | None = None,
    ) -> None:
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.file_manager = file_manager or LammpsFileManager(config)
        self.executor = executor or LammpsExecutor()
        self.parser = parser or LammpsResultParser(config)

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

    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        with ctx:
            # Generate input script to file
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(f, potential_path, data_file, dump_file, elements)

            # Initialize Driver with unique log file
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            try:
                self.executor.execute_simulation(driver, input_script_path)
                return self.parser.parse_md_result(driver, dump_file, log_file)
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
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self.executor.execute_simulation(driver, script_path)
                return driver.get_atoms(elements)
            finally:
                if hasattr(driver, "close"):
                    driver.close()
