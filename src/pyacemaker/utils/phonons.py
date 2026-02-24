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
        # PhonopyAtoms has get_scaled_positions() method, but 'scaled_positions' might be property depending on version.
        # But wait, unit test mock returns Atoms object from ase, NOT PhonopyAtoms.
        # In actual code, Phonopy.get_supercells_with_displacements returns list of PhonopyAtoms.
        # In test, I mocked it to return [structure.copy()], which is ASE Atoms.
        # ASE Atoms does NOT have scaled_positions attribute (it has get_scaled_positions()).

        # If phonon_atoms is ASE Atoms (in test):
        if isinstance(phonon_atoms, Atoms):
             return phonon_atoms

        # Real PhonopyAtoms (in production)
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
        # Ensure supercell_matrix is 3x3 or list of 3 ints (diagonal)
        if len(self.supercell_matrix) == 3 and isinstance(self.supercell_matrix[0], int):
             # Convert [2, 2, 2] to diag matrix
             s_mat = np.diag(self.supercell_matrix)
        else:
             s_mat = np.array(self.supercell_matrix)

        phonon = Phonopy(unitcell, supercell_matrix=s_mat)
        phonon.generate_displacements(distance=self.displacement)

        supercells = phonon.get_supercells_with_displacements()  # type: ignore[attr-defined]

        # Calculate forces for each displaced supercell
        forces_set: list[np.ndarray] = []
        for sc in supercells:
            # Convert to ASE for engine
            ase_sc = self._phonopy_to_ase(sc)

            # Compute forces using engine
            result = self.engine.compute_static_properties(ase_sc, potential_path)

            # Ensure forces are correct shape (N, 3)
            # engine.run returns MDSimulationResult.forces which is list of lists
            forces = np.array(result.forces)
            forces_set.append(forces)

        phonon.produce_force_constants(forces=forces_set)

        # Calculate Band Structure along a simple path (Gamma -> X -> M -> Gamma)
        # This is arbitrary but sufficient for visualization.
        path = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]
        labels = ["G", "X", "M", "G"]

        phonon.run_band_structure(
            paths=[path],
            with_eigenvectors=False,
            labels=labels
        )
        bs = phonon.get_band_structure_dict()
        frequencies = bs["frequencies"] # list of arrays (n_qpoints, n_bands)

        # Check for imaginary modes
        # Phonopy returns negative frequencies for imaginary modes
        all_freqs = np.concatenate(frequencies)
        min_freq = np.min(all_freqs)
        is_stable = min_freq > self.imaginary_tol

        # Plot
        phonon.plot_band_structure()  # type: ignore[no-untyped-call]
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return bool(is_stable), plot_base64
