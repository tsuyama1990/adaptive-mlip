import base64
import io
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    HAS_PHONOPY = True
else:
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        HAS_PHONOPY = True
    except ImportError:
        HAS_PHONOPY = False
        Phonopy = Any
        PhonopyAtoms = Any

def plot_to_base64(fig: Figure) -> str:
    """Encodes a matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data

class PhononCalculator:
    """
    Calculates phonon properties using Phonopy.
    """

    def __init__(self, atoms: Atoms, supercell_matrix: list[int] | None = None) -> None:
        """
        Args:
            atoms: Unit cell structure.
            supercell_matrix: Supercell matrix (e.g. [2, 2, 2]).
        """
        if not HAS_PHONOPY:
            msg = "Phonopy is not installed."
            raise ImportError(msg)

        self.atoms = atoms
        self.supercell_matrix = supercell_matrix or [2, 2, 2]
        self.phonopy: Phonopy | None = None
        self._setup_phonopy()

    def _setup_phonopy(self) -> None:
        """Initializes the Phonopy object."""
        # Conversion from ASE to PhonopyAtoms
        unitcell = PhonopyAtoms(
            symbols=self.atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            cell=self.atoms.get_cell(),  # type: ignore[no-untyped-call]
            scaled_positions=self.atoms.get_scaled_positions()  # type: ignore[no-untyped-call]
        )

        # Determine supercell matrix format
        # If list [2,2,2], convert to diagonal matrix
        if len(self.supercell_matrix) == 3:
            smat = np.diag(self.supercell_matrix)
        else:
            smat = np.array(self.supercell_matrix)

        self.phonopy = Phonopy(unitcell, supercell_matrix=smat)

    def calculate_forces(self, force_fn: Callable[[Atoms], list[list[float]]]) -> None:
        """
        Calculates forces for supercells with displacements.

        Args:
            force_fn: A function that takes an ASE Atoms object and returns forces (Nx3 list/array).
        """
        if self.phonopy is None:
            msg = "Phonopy not initialized"
            raise RuntimeError(msg)

        # Generate displacements
        self.phonopy.generate_displacements(distance=0.01)

        supercells = self.phonopy.supercells_with_displacements
        # sets of forces
        forces_set = []

        if supercells is not None:
            for sc in supercells:
                # Convert PhonopyAtoms back to ASE
                # Use attributes directly if getters fail
                symbols = sc.get_chemical_symbols() if hasattr(sc, "get_chemical_symbols") else sc.symbols
                positions = sc.get_scaled_positions() if hasattr(sc, "get_scaled_positions") else sc.scaled_positions
                cell = sc.get_cell() if hasattr(sc, "get_cell") else sc.cell

                ase_sc = Atoms(
                    symbols=symbols,
                    scaled_positions=positions,
                    cell=cell,
                    pbc=True
                )

                # Calculate forces using the provided function (e.g. LAMMPS calculator)
                f = force_fn(ase_sc)
                forces_set.append(f)

        self.phonopy.forces = forces_set
        self.phonopy.produce_force_constants()

    def check_stability(self, tolerance: float = -0.05) -> tuple[bool, list[float]]:
        """
        Checks for dynamical stability (imaginary frequencies).

        Args:
            tolerance: Tolerance for imaginary frequencies (in THz).
                       Phonopy uses negative values for imaginary frequencies.
                       Frequencies < tolerance are considered unstable.
                       e.g. -0.05 means we allow small numerical noise down to -0.05 THz.

        Returns:
            (is_stable, imaginary_frequencies)
        """
        if self.phonopy is None:
            msg = "Phonopy not initialized"
            raise RuntimeError(msg)

        # Calculate mesh to sample frequencies
        mesh = [20, 20, 20]
        self.phonopy.run_mesh(mesh)
        mesh_dict = self.phonopy.get_mesh_dict()
        frequencies = mesh_dict["frequencies"] # (q-points, bands)

        # Flatten
        all_freqs = frequencies.flatten()

        # Filter imaginary (negative)
        # Usually phonopy returns negative real numbers for imaginary modes
        imaginary = [f for f in all_freqs if f < tolerance]

        is_stable = len(imaginary) == 0
        return is_stable, imaginary

    def get_band_structure_plot(self) -> str:
        """Generates band structure plot as base64 string."""
        if self.phonopy is None:
            msg = "Phonopy not initialized"
            raise RuntimeError(msg)

        # Auto band path
        self.phonopy.auto_band_structure(plot=True)  # type: ignore[no-untyped-call]
        self.phonopy.plot_band_structure()  # type: ignore[no-untyped-call]

        # plot is a matplotlib pyplot module or figure?
        # Phonopy's plot_band_structure returns plt.
        # So we get the current figure.
        fig = plt.gcf()
        return plot_to_base64(fig)

    def get_dos_plot(self) -> str:
        """Generates DOS plot as base64 string."""
        if self.phonopy is None:
            msg = "Phonopy not initialized"
            raise RuntimeError(msg)

        self.phonopy.run_total_dos()
        self.phonopy.plot_total_dos()
        fig = plt.gcf()
        return plot_to_base64(fig)
