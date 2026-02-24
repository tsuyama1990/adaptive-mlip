from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.exceptions import EngineError
from pyacemaker.core.input_validation import LammpsValidator
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
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
        file_manager: LammpsFileManager | None = None,
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

        Args:
            structure: Input structure (Atoms object)
            potential: Path to potential file (str or Path)

        Returns:
            MDSimulationResult containing trajectory and energy data.

        Raises:
            EngineError: If simulation fails.
        """
        try:
            # Input Validation (SRP via Validator)
            LammpsValidator.validate_structure(structure)
            # Strict typing enforcement for potential
            if not isinstance(potential, (str, Path)):
                msg = f"Potential must be a path (str or Path), got {type(potential)}"
                raise TypeError(msg)

            potential_path = LammpsValidator.validate_potential(potential)

            # Ensure potential path is resolved and exists (double check for race conditions/symlinks)
            potential_path = potential_path.resolve(strict=True)

            # Prepare workspace (temp dir, file writing)
            if structure is None:
                msg = "Structure cannot be None after validation."
                raise ValueError(msg)

            ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(
                structure
            )

            with ctx:
                input_script_path = data_file.parent / "input.lmp"

                with input_script_path.open("w") as f:
                    self.generator.write_script(
                        f, potential_path.resolve(), data_file, dump_file, elements
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
                    except Exception:
                        energy = 0.0
                        temperature = 0.0
                        step = 0

                    max_gamma = 0.0
                    if self.config.fix_halt:
                        try:
                            max_gamma = driver.extract_variable("max_g")
                        except Exception:
                            max_gamma = 0.0

                    halted = False
                    if self.config.fix_halt:
                        halted = step < self.config.n_steps

                    # Result
                    return MDSimulationResult(
                        energy=energy,
                        forces=[[0.0, 0.0, 0.0]],
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

        except Exception as e:
            if isinstance(e, (TypeError, ValueError, FileNotFoundError)):
                # Re-raise validation errors directly
                raise
            # Wrap others in EngineError
            msg = f"Engine execution failed: {e}"
            raise EngineError(msg) from e

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        """Executes the simulation script with error handling."""
        try:
            driver.run(script_path.read_text())
        except RuntimeError as e:
            msg = f"LAMMPS execution failed: {e}"
            raise EngineError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during LAMMPS execution: {e}"
            raise EngineError(msg) from e
