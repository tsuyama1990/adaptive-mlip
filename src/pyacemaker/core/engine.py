from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
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
        if structure is None:
             msg = "Structure must be provided."
             raise ValueError(msg)

        if len(structure) == 0:
             msg = "Structure contains no atoms."
             raise ValueError(msg)

        # Prepare workspace (temp dir, file writing)
        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(structure)

        with ctx:
            # Generate script using delegate
            # Note: potential is passed as is. Validation of potential existence
            # should ideally happen, but LammpsFileManager handles data file.
            # LammpsEngine should check potential if it's a path.
            # (Adding check similar to previous implementation)
            # Actually, io_manager doesn't handle potential file logic.

            # Check potential path
            # We assume potential is a path string or Path object.
            # LammpsDriver needs string.
            from pathlib import Path
            potential_path = Path(potential)
            if not potential_path.exists():
                 msg = f"Potential file not found: {potential_path}"
                 raise FileNotFoundError(msg)

            script = self.generator.generate(
                potential_path.resolve(),
                data_file,
                dump_file,
                elements
            )

            # Initialize Driver with unique log file
            driver = LammpsDriver(["-screen", "none", "-log", str(log_file)])

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
                max_gamma = driver.extract_variable("max_g")
            except Exception:
                energy = 0.0
                temperature = 0.0
                step = 0
                max_gamma = 0.0

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
