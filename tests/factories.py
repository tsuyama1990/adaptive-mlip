import numpy as np
from ase import Atoms


def create_mock_atoms(
    element: str = "H",
    cell: list[float] | None = None,
    pbc: bool = True,
    energy: float = -10.0,
    max_gamma: float = 0.1,
) -> Atoms:
    """
    Factory function to create a mocked ASE Atoms object with prepopulated info and arrays.
    Useful for testing MACE and Oracle interactions without real calculation dependencies.
    """
    if cell is None:
        cell = [10.0, 10.0, 10.0]

    atoms = Atoms(element, cell=cell, pbc=pbc)

    # Pre-populate Mock Properties
    atoms.info["energy"] = energy
    atoms.new_array("forces", np.zeros((len(atoms), 3)))

    # Optional gamma logic
    c_gamma = np.random.uniform(0.01, max_gamma, size=len(atoms))
    atoms.new_array("c_gamma", c_gamma)

    return atoms
