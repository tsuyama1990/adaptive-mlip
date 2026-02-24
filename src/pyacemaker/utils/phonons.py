import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToTHz

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.validation import (
    PhononConfig,
    PhononResult,
    ValidationStatus,
)
from pyacemaker.utils.lammps import run_static_lammps

logger = logging.getLogger(__name__)


class PhononCalculator:
    """Calculates phonon properties and checks stability using Phonopy and LAMMPS."""

    def __init__(self, config: PhononConfig, md_config: MDConfig) -> None:
        self.config = config
        self.md_config = md_config

    def _ase_to_phonopy(self, atoms: Atoms) -> PhonopyAtoms:
        """Converts ASE Atoms to PhonopyAtoms."""
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            scaled_positions=atoms.get_scaled_positions(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell()[:],  # type: ignore[no-untyped-call]
        )

    def calculate(
        self, structure: Atoms, potential_path: Path, output_dir: Path
    ) -> PhononResult:
        """
        Runs phonon calculation.

        Args:
            structure: Unit cell (ASE Atoms).
            potential_path: Path to potential.
            output_dir: Directory to save plots.

        Returns:
            PhononResult object.
        """
        unit_cell = self._ase_to_phonopy(structure)
        supercell_matrix = list(self.config.supercell_size)

        phonon = Phonopy(
            unit_cell,
            supercell_matrix,
            symprec=self.config.symprec,
            factor=VaspToTHz,
        )

        phonon.generate_displacements(distance=self.config.displacement)
        supercells = phonon.supercells_with_displacements

        if supercells is None:
            msg = "Failed to generate supercells."
            raise RuntimeError(msg)

        forces_list = []
        for ph_sc in supercells:
            # Convert PhonopyAtoms back to ASE Atoms for LAMMPS
            ase_sc = Atoms(
                symbols=ph_sc.symbols,
                scaled_positions=ph_sc.scaled_positions,
                cell=ph_sc.cell,
                pbc=True,
            )

            # Run static calculation
            _, forces, _ = run_static_lammps(ase_sc, potential_path, self.md_config)
            forces_list.append(forces)

        # Set forces
        phonon.produce_force_constants(forces=forces_list)

        # Check for imaginary modes at Gamma point or along path
        # Simple check: calculate mesh and check frequencies
        mesh = [20, 20, 20]
        phonon.run_mesh(mesh, with_eigenvectors=False, is_mesh_symmetry=False)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict["frequencies"]  # (q-points, bands)

        # Imaginary frequencies are negative in phonopy
        min_freq = np.min(frequencies)
        has_imaginary = min_freq < -0.05  # Tolerance for numerical noise (0.05 THz)

        # Generate Band Structure Plot
        path_bands = output_dir / "phonon_bands.png"
        self._plot_bands(phonon, path_bands)

        status = ValidationStatus.FAIL if has_imaginary else ValidationStatus.PASS

        return PhononResult(
            has_imaginary_modes=has_imaginary,
            band_structure_path=path_bands,
            status=status,
        )

    def _plot_bands(self, phonon: Phonopy, output_path: Path) -> None:
        """Generates band structure plot."""
        try:
            phonon.auto_band_structure(plot=True).savefig(output_path)  # type: ignore[no-untyped-call]
            plt.close()
        except Exception:
            logger.exception("Failed to plot band structure")
