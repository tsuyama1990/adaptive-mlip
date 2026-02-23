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

    def _generate_input_script(
        self,
        structure: Atoms,
        potential_path: str,
        data_file: str,
        dump_file: str,
        elements: list[str],
    ) -> str:
        """
        Generates the LAMMPS input script.
        """
        # 1. Basics
        lines = [
            "clear",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"read_data {data_file}",
        ]

        # 2. Potential
        # Ensure elements match the order in data file (specorder)
        species_str = " ".join(elements)

        if self.config.hybrid_potential:
            # Hybrid overlay: PACE + ZBL
            params = self.config.hybrid_params
            lines.append(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}")

            # PACE
            lines.append(f"pair_coeff * * pace {potential_path} {species_str}")

            # ZBL
            # Generate pair_coeff for all pairs (i, j) based on atomic numbers
            # LAMMPS types are 1-based, corresponding to elements list order
            n_types = len(elements)
            for i in range(n_types):
                el_i = elements[i]
                z_i = atomic_numbers[el_i]
                for j in range(i, n_types):
                    # For ZBL, pair_coeff I J zbl Z_I Z_J
                    el_j = elements[j]
                    z_j = atomic_numbers[el_j]
                    lines.append(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}")
        else:
            # Pure PACE
            lines.append("pair_style pace")
            lines.append(f"pair_coeff * * pace {potential_path} {species_str}")

        # 3. Settings
        lines.append("neighbor 2.0 bin")
        lines.append("neigh_modify delay 0 every 1 check yes")
        lines.append(f"timestep {self.config.timestep}")

        # 4. Compute / Fix Halt (Uncertainty Watchdog)
        # compute pace returns per-atom gamma
        lines.append(f"compute gamma all pace {potential_path}")
        lines.append("compute max_gamma all reduce max c_gamma")
        lines.append("variable max_g equal c_max_gamma")

        # fix halt
        # Check every check_interval steps
        # If max_g > threshold, stop.
        # error continue -> stop run but continue script
        lines.append(
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue"
        )

        # 5. MD / Minimization
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

        # 6. Output
        lines.append(f"thermo {self.config.thermo_freq}")
        lines.append("thermo_style custom step temp pe press v_max_g")
        lines.append(f"dump traj all custom {self.config.dump_freq} {dump_file} id type x y z c_gamma")

        # 7. Run
        lines.append(f"run {self.config.n_steps}")

        # 8. Post-Run
        # Check if halted.
        lines.append("variable halted equal f_halt_check")
        lines.append("print 'Halted: ${halted}'")

        return "\n".join(lines)

    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        if structure is None:
             msg = "Structure must be provided."
             raise ValueError(msg)

        # Use temporary directory for intermediate files to prevent clutter/race conditions
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            run_id = uuid.uuid4().hex[:8]

            # Intermediate files (in temp)
            data_file = tmp_dir / f"data_{run_id}.lmp"

            # Output files (in CWD for persistence)
            # We want to return paths to files that persist after temp dir cleanup.
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
            # Note: LAMMPS script needs paths.
            # If we pass absolute paths, it works regardless of CWD.
            # data_file is absolute (tmp_dir).
            # dump_file is absolute (cwd).
            # potential_path might be relative. We should resolve it?
            # Or assume LAMMPS runs in CWD?
            # LammpsDriver runs in current process, so CWD is same.

            script = self._generate_input_script(
                structure, str(potential), str(data_file), str(dump_file), elements
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
                # If variables not available, return defaults
                energy = 0.0
                temperature = 0.0
                step = 0
                max_gamma = 0.0

            halted = step < self.config.n_steps

            # Result
            return MDSimulationResult(
                energy=energy,
                forces=[[0.0, 0.0, 0.0]], # Placeholder as we dump trajectory
                halted=halted,
                max_gamma=max_gamma,
                n_steps=step,
                temperature=temperature,
                trajectory_path=str(dump_file),
                log_path=str(log_file),
                halt_structure_path=str(dump_file) if halted else None
            )
