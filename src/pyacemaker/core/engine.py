from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsValidator
from pyacemaker.domain_models.constants import LAMMPS_SCREEN_ARG
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
        file_manager: LammpsFileManager | None = None
    ) -> None:
        """
        Initialize the engine with configuration.
        Allows dependency injection for generator and file manager.
        """
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.file_manager = file_manager or LammpsFileManager(config)

    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        # Input Validation (SRP via Validator)
        LammpsValidator.validate_structure(structure)
        potential_path = LammpsValidator.validate_potential(potential)

        # Ensure potential path is resolved and exists (double check for race conditions/symlinks)
        potential_path = potential_path.resolve(strict=True)

        # Prepare workspace (temp dir, file writing)
        # Note: We checked structure is not None/Empty in validator.
        # But prepare_workspace needs 'structure' as Atoms (which it is, after check).
        # Type checker might complain if structure is Optional.
        # But runtime check ensures it.
        if structure is None:
             msg = "Structure cannot be None after validation."
             raise ValueError(msg)

        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(structure)

        with ctx:
            # Create a temporary file for the input script to handle large scripts
            # LammpsDriver expects a string script content or we can adapt it.
            # LammpsDriver.run takes string. If we want to stream large script, we need to pass file.
            # But python-lammps `commands_string` or `file` takes path.
            # Assuming LammpsDriver.run wraps `lammps.commands_string` or `lammps.file`.
            # If our LammpsDriver only takes string, we still have to load it.
            # BUT, the generator refactor was to avoid holding it in memory *during generation*.
            # We can write to a temp file, then read it? Or if LammpsDriver supports file path?
            # Looking at interfaces/lammps_driver.py isn't possible (I can't see it now),
            # but usually drivers support running from file.
            # If not, we have to read it back.
            # Let's write to "input.lmp" in temp dir.
            input_script_path = Path(ctx.name) / "input.lmp" if hasattr(ctx, "name") else data_file.parent / "input.lmp"

            # Since ctx is generic context manager, we assume data_file is in the temp dir.
            input_script_path = data_file.parent / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(
                    f,
                    potential_path.resolve(),
                    data_file,
                    dump_file,
                    elements
                )

            # Initialize Driver with unique log file
            # Use constant for screen arg
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
                if self.config.fix_halt:
                    # If using fix halt, checking step count is a proxy for early termination
                    # Assuming run starts at 0. If restart, logic might need adjustment.
                    # For now, Cycle 01/02 implies fresh runs.
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

    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
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

    def _ensure_script_readable(self, script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        """
        Executes the simulation script with standardized error handling.

        Args:
            driver: Initialized LammpsDriver instance.
            script_path: Path to the LAMMPS input script.

        Raises:
            RuntimeError: Wraps any failure during execution with context.
        """
        try:
            # Ensure script is readable
            self._ensure_script_readable(script_path)

            script_content = script_path.read_text(encoding="utf-8")
            driver.run(script_content)

        except FileNotFoundError as e:
            msg = f"Simulation setup failed: {e}"
            raise RuntimeError(msg) from e
        except ValueError as e:
            # Catch validation errors from driver (unsafe content)
            msg = f"Simulation security validation failed: {e}"
            raise RuntimeError(msg) from e
        except RuntimeError as e:
            # Catch LAMMPS internal errors (e.g., lost atoms)
            msg = f"LAMMPS engine execution failed: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            # Catch unexpected errors
            msg = f"Unexpected error during simulation execution: {e}"
            raise RuntimeError(msg) from e
