import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.interfaces.lammps_driver import LammpsDriver
from pyacemaker.utils.structure import get_species_order

logger = logging.getLogger(__name__)


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

        if len(structure) > 10000:
             logger.warning("Simulating large structure (%d atoms). Memory usage may be high.", len(structure))

        # Validate potential path
        potential_path = Path(potential)
        if not potential_path.exists():
             msg = f"Potential file not found: {potential_path}"
             raise FileNotFoundError(msg)

        # Use temporary directory (RAM disk if possible via config)
        with tempfile.TemporaryDirectory(dir=self.config.temp_dir) as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            run_id = uuid.uuid4().hex[:8]

            # Intermediate files (in temp)
            data_file = tmp_dir / f"data_{run_id}.lmp"

            # Output files (in CWD for persistence)
            cwd = Path.cwd()
            dump_file = cwd / f"dump_{run_id}.lammpstrj"
            log_file = cwd / f"log_{run_id}.lammps"

            # Get elements for specorder
            elements = get_species_order(structure)

            # Write structure to LAMMPS data file in temp dir
            try:
                write(str(data_file), structure, format="lammps-data", specorder=elements, atom_style="atomic")
            except Exception as e:
                msg = f"Failed to write LAMMPS data file: {e}"
                raise RuntimeError(msg) from e

            # Generate script using delegate
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
