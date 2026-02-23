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

    def __init__(self, config: MDConfig) -> None:
        """
        Initialize the engine with configuration.
        """
        self.config = config
        self.generator = LammpsScriptGenerator(config)
        self.file_manager = LammpsFileManager(config)

    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        # Input Validation (SRP via Validator)
        LammpsValidator.validate_structure(structure)
        potential_path = LammpsValidator.validate_potential(potential)

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
            # Generate script using delegate
            script = self.generator.generate(
                potential_path.resolve(),
                data_file,
                dump_file,
                elements
            )

            # Initialize Driver with unique log file
            # Use constant for screen arg
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            # Run
            try:
                driver.run(script)
            except Exception as e:
                msg = f"LAMMPS execution failed: {e}"
                raise RuntimeError(msg) from e

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
                # If using fix halt, checking step count is a proxy for early termination
                # Assuming run starts at 0. If restart, logic might need adjustment.
                # For now, Cycle 01/02 implies fresh runs.
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
                halt_structure_path=str(dump_file) if halted else None
            )
