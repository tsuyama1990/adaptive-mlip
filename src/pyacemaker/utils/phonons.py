import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.domain_models.defaults import DEFAULT_VALIDATION_PHONON_IMAGINARY_TOL


class PhononCalculator:
    """
    Calculates phonon band structures and checks for dynamical stability using Phonopy.
    """

    def __init__(
        self,
        engine: BaseEngine,
        supercell_matrix: list[int],
        displacement: float,
        imaginary_tol: float = DEFAULT_VALIDATION_PHONON_IMAGINARY_TOL,
    ) -> None:
        self.engine = engine
        self.supercell_matrix = supercell_matrix
        self.displacement = displacement
        self.imaginary_tol = imaginary_tol

    def _ase_to_phonopy(self, atoms: Atoms) -> PhonopyAtoms:
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell(),  # type: ignore[no-untyped-call]
            scaled_positions=atoms.get_scaled_positions(),  # type: ignore[no-untyped-call]
        )

    def _phonopy_to_ase(self, phonon_atoms: PhonopyAtoms) -> Atoms:
        if isinstance(phonon_atoms, Atoms):
             return phonon_atoms

        return Atoms(
            symbols=phonon_atoms.symbols,
            cell=phonon_atoms.cell,
            scaled_positions=phonon_atoms.scaled_positions,
            pbc=True,
        )

    def check_stability(self, structure: Atoms, potential_path: Path) -> tuple[bool, str]:
        """
        Calculates phonons and checks for imaginary modes.
        Returns stability status and base64 encoded band structure plot.
        """
        unitcell = self._ase_to_phonopy(structure)
        if len(self.supercell_matrix) == 3 and isinstance(self.supercell_matrix[0], int):
             s_mat = np.diag(self.supercell_matrix)
        else:
             s_mat = np.array(self.supercell_matrix)

        phonon = Phonopy(unitcell, supercell_matrix=s_mat)
        phonon.generate_displacements(distance=self.displacement)

        # Optimization:
        # Although phonopy.get_supercells_with_displacements() returns a list,
        # we process and discard ASE objects immediately to minimize overhead.
        supercells = phonon.get_supercells_with_displacements()  # type: ignore[attr-defined]

        forces_set: list[np.ndarray] = []

        # Iterate and process
        for i, sc in enumerate(supercells):
            # Convert to ASE
            ase_sc = self._phonopy_to_ase(sc)

            # Compute forces
            result = self.engine.compute_static_properties(ase_sc, potential_path)

            # Store only forces, discard ase_sc
            forces = np.array(result.forces)
            forces_set.append(forces)

            # Explicitly delete ASE object to help GC in tight loops
            del ase_sc

        phonon.produce_force_constants(forces=forces_set)

        path = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
        labels = ["G", "X", "M", "G"]

        phonon.run_band_structure(
            paths=[path],
            with_eigenvectors=False,
            labels=labels
        )
        bs = phonon.get_band_structure_dict()
        frequencies = bs["frequencies"]

        all_freqs = np.concatenate(frequencies)
        min_freq = np.min(all_freqs)
        is_stable = min_freq > self.imaginary_tol

        phonon.plot_band_structure()  # type: ignore[no-untyped-call]
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return bool(is_stable), plot_base64
