import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write

from pyacemaker.core.base import BaseEngine
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

        # Use RAM disk (/dev/shm) for temporary files if available to reduce I/O latency
        self._temp_root: str | None = None
        shm_path = Path("/dev/shm")  # noqa: S108
        if shm_path.exists() and shm_path.is_dir() and os.access(shm_path, os.W_OK):
            self._temp_root = str(shm_path)

    def _gen_potential(self, potential_path: Path, elements: list[str]) -> list[str]:
        """Generates potential definition commands."""
        lines = []
        species_str = " ".join(elements)

        if self.config.hybrid_potential:
            # Hybrid overlay: PACE + ZBL
            params = self.config.hybrid_params
            lines.append(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}")

            # PACE
            lines.append(f"pair_coeff * * pace {potential_path} {species_str}")

            # ZBL
            # Generate pair_coeff for all pairs (i, j) based on atomic numbers
            n_types = len(elements)
            for i in range(n_types):
                el_i = elements[i]
                z_i = atomic_numbers[el_i]
                for j in range(i, n_types):
                    el_j = elements[j]
                    z_j = atomic_numbers[el_j]
                    lines.append(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}")
        else:
            # Pure PACE
            lines.append("pair_style pace")
            lines.append(f"pair_coeff * * pace {potential_path} {species_str}")

        return lines

    def _gen_settings(self) -> list[str]:
        """Generates general MD settings."""
        lines = []
        lines.append(f"neighbor {self.config.neighbor_skin} bin")
        lines.append("neigh_modify delay 0 every 1 check yes")
        lines.append(f"timestep {self.config.timestep}")
        return lines

    def _gen_watchdog(self, potential_path: Path) -> list[str]:
        """Generates Uncertainty Watchdog commands."""
        lines = []
        lines.append(f"compute gamma all pace {potential_path}")
        lines.append("compute max_gamma all reduce max c_gamma")
        lines.append("variable max_g equal c_max_gamma")

        lines.append(
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue"
        )
        return lines

    def _gen_execution(self) -> list[str]:
        """Generates minimization and MD run commands."""
        lines = []

        if self.config.minimize:
            lines.append("minimize 1.0e-4 1.0e-6 100 1000")

        # Calculate damping parameters (recommended: Tdamp ~ 100*dt, Pdamp ~ 1000*dt)
        tdamp = 100.0 * self.config.timestep
        pdamp = 1000.0 * self.config.timestep

        lines.append(f"velocity all create {self.config.temperature} 12345")
        lines.append(
            f"fix npt all npt temp {self.config.temperature} {self.config.temperature} {tdamp} "
            f"iso {self.config.pressure} {self.config.pressure} {pdamp}"
        )

        lines.append(f"run {self.config.n_steps}")
        return lines

    def _gen_output(self, dump_file: Path) -> list[str]:
        """Generates output settings."""
        lines = []
        lines.append(f"thermo {self.config.thermo_freq}")
        lines.append("thermo_style custom step temp pe press v_max_g")
        lines.append(f"dump traj all custom {self.config.dump_freq} {dump_file} id type x y z c_gamma")

        # Check if halted
        lines.append("variable halted equal f_halt_check")
        lines.append("print 'Halted: ${halted}'")
        return lines

    def _generate_input_script(
        self,
        potential_path: Path,
        data_file: Path,
        dump_file: Path,
        elements: list[str],
    ) -> str:
        """
        Orchestrates LAMMPS input script generation.
        """
        lines = [
            "clear",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"read_data {data_file}",
        ]

        lines.extend(self._gen_potential(potential_path, elements))
        lines.extend(self._gen_settings())
        lines.extend(self._gen_watchdog(potential_path))
        lines.extend(self._gen_execution())
        lines.extend(self._gen_output(dump_file))

        return "\n".join(lines)

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

        # Use temporary directory (RAM disk if possible)
        with tempfile.TemporaryDirectory(dir=self._temp_root) as tmp_dir_str:
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

            # Generate script
            script = self._generate_input_script(
                potential_path.resolve(), # Use absolute path
                data_file,
                dump_file,
                elements
            )

            # Initialize Driver
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
