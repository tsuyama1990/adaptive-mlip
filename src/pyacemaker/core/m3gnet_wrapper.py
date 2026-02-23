from ase import Atoms
from ase.build import bulk


class M3GNetWrapper:
    """
    Wrapper for M3GNet structure predictor.
    Currently implements a mock strategy using ASE build tools.
    """

    def predict_structure(self, composition: str) -> Atoms:
        """
        Predicts a stable structure for the given composition.

        Args:
            composition: Chemical formula (e.g., "Fe", "FePt").

        Returns:
            ASE Atoms object representing the predicted structure.
        """
        try:
            # Try to build a bulk structure for simple elements
            # 'cubic=True' returns a conventional cell which is better for defects/supercells
            return bulk(composition, cubic=True)
        except Exception:
            # If ase.build.bulk fails (e.g. for complex alloys like FePt that it doesn't know),
            # we construct a simple dummy lattice.
            atoms = Atoms(composition)
            # Create a simple cubic box
            # Estimate volume based on number of atoms (approx 10-15 A^3 per atom)
            vol_per_atom = 15.0
            L = (len(atoms) * vol_per_atom) ** (1 / 3)
            atoms.set_cell([L, L, L])  # type: ignore[no-untyped-call]
            atoms.set_pbc(True)  # type: ignore[no-untyped-call]

            # Arrange atoms in a grid or random?
            # Random is safer for "Cold Start" if we don't know structure

            # Quick hack: spread them out slightly.
            import numpy as np

            pos = np.random.uniform(0, 1, (len(atoms), 3))
            atoms.set_scaled_positions(pos)  # type: ignore[no-untyped-call]

            return atoms
