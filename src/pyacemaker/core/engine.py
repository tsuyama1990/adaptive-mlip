import logging
from pathlib import Path

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.constants import (
    ERR_SIM_EXEC_FAIL,
    ERR_SIM_SECURITY_FAIL,
    ERR_SIM_SETUP_FAIL,
    LAMMPS_SCREEN_ARG,
)
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.interfaces.lammps_driver import LammpsDriver

logger = logging.getLogger(__name__)

class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS.
    Handles input generation, execution, and result parsing.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator | None = None,
        file_manager: LammpsFileManager | None = None
    ) -> None:
        """
        Initialize the engine with configuration.
        Allows dependency injection for generator and file manager.
        """
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.file_manager = file_manager or LammpsFileManager(config)

    def _ensure_script_readable(self, script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    def _verify_sandbox(self, script_path: Path) -> None:
        content = script_path.read_text()
        forbidden_commands = ["shell ", "include ", "read_restart "]
        for cmd in forbidden_commands:
            if cmd in content:
                msg = f"Forbidden command '{cmd}' found in execution payload."
                raise ValueError(msg)

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        """
        Executes the simulation script with standardized error handling,
        incorporating sandboxing heuristics and resource limits to prevent DoS.
        """
        self._ensure_script_readable(script_path)
        self._verify_sandbox(script_path)

        try:
            import concurrent.futures
            # Scalability: Use run_file to stream script execution within a timeout wrapper
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(driver.run_file, str(script_path))
                # Enforce resource boundary limit of 600s (10 min) per script execution
                future.result(timeout=600)

        except concurrent.futures.TimeoutError as e:
             msg = "LAMMPS simulation exceeded hard resource execution limits (600s). Aborting."
             raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            raise RuntimeError(ERR_SIM_SETUP_FAIL.format(error=e)) from e
        except ValueError as e:
            raise RuntimeError(ERR_SIM_SECURITY_FAIL.format(error=e)) from e
        except RuntimeError as e:
            raise RuntimeError(ERR_SIM_EXEC_FAIL.format(error=e)) from e

    def run(self, structure: Atoms | None, potential: str | Path | None) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        from pyacemaker.domain_models.constants import ERR_STRUCTURE_NONE
        if structure is None:
             raise ValueError(ERR_STRUCTURE_NONE)
        ctx, data_file, dump_file, log_file, elements, potential_path = self.file_manager.prepare_workspace(structure, potential)

        with ctx:
            # Generate input script to file
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(
                    f,
                    potential_path,
                    data_file,
                    dump_file,
                    elements
                )

            # Initialize Driver with unique log file
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            try:
                self._execute_simulation(driver, input_script_path)

                # Extract Results
                try:
                    energy = driver.extract_variable("pe")
                    temperature = driver.extract_variable("temp")
                    step = int(driver.extract_variable("step"))
                    forces = driver.get_forces().tolist()
                    stress = driver.get_stress().tolist()
                except (ValueError, KeyError, AttributeError) as e:
                    msg = f"Failed to extract mandatory properties from simulation driver: {e}"
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

                max_gamma = 0.0
                if self.config.fix_halt:
                    try:
                        max_gamma = driver.extract_variable("max_g")
                    except (ValueError, KeyError, AttributeError) as e:
                        logger.warning("Uncertainty metrics ('max_g') requested but unavailable: %s", e)
                        max_gamma = 0.0

                halted = False
                if self.config.fix_halt:
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
                    halt_step=step if halted else None
                )
            finally:
                if hasattr(driver, "close"):
                    driver.close()

    def compute_static_properties(self, structure: Atoms, potential: str | Path | None) -> MDSimulationResult:
        """
        Computes static properties (energy, forces, stress) for a structure.
        Equivalent to a 0-step MD run.
        """
        static_config = self.config.model_copy(update={
            "n_steps": 0,
            "minimize": False,
            "thermo_freq": 1,
            "dump_freq": 0
        })

        engine = LammpsEngine(static_config)
        return engine.run(structure, potential)

    def relax(self, structure: Atoms | None, potential: str | Path | None) -> Atoms:
        """
        Relaxes the structure to a local minimum using LAMMPS minimize.
        """
        from pyacemaker.domain_models.constants import ERR_STRUCTURE_NONE
        if structure is None:
             raise ValueError(ERR_STRUCTURE_NONE)
        ctx, data_file, dump_file, log_file, elements, potential_path = self.file_manager.prepare_workspace(structure, potential)

        with ctx:
            # Generate minimization script
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            script_path = temp_dir / "relax.lmp"

            with script_path.open("w") as f:
                self.generator.write_minimization_script(
                    f,
                    potential_path,
                    data_file,
                    elements
                )

            # Execute
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self._execute_simulation(driver, script_path)
                return driver.get_atoms(elements)
            finally:
                 if hasattr(driver, "close"):
                     driver.close()
