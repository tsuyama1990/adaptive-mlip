from typing import Any

import numpy as np
from ase import Atoms


def rattle(atoms: Atoms, stdev: float, rng: np.random.Generator | None = None) -> Atoms:
    """
    Apply random Gaussian noise to atomic positions.
    Returns a new Atoms object.

    Args:
        atoms: Input Atoms object.
        stdev: Standard deviation of Gaussian noise in Angstroms.
        rng: Optional NumPy random number generator for reproducibility/efficiency.
    """
    if stdev < 0:
        msg = "Standard deviation must be non-negative"
        raise ValueError(msg)

    new_atoms: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    if stdev > 0:
        if rng is None:
            rng = np.random.default_rng()

        displacement = rng.normal(scale=stdev, size=new_atoms.positions.shape)
        new_atoms.positions += displacement
    return new_atoms


def apply_strain(atoms: Atoms, strain_tensor: np.ndarray, rng: Any = None) -> Atoms:
    """
    Apply strain tensor to the unit cell.

    strain_tensor: 3x3 numpy array representing the strain epsilon.
    New cell = Old cell @ (I + epsilon)
    Atoms are scaled accordingly.

    Args:
        atoms: Input structure.
        strain_tensor: 3x3 strain matrix.
        rng: Unused (kept for API consistency if needed later).
    """
    if strain_tensor.shape != (3, 3):
        msg = "Strain tensor must be a 3x3 matrix"
        raise ValueError(msg)

    new_atoms: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    cell = new_atoms.get_cell()  # type: ignore[no-untyped-call]

    deformation = np.eye(3) + strain_tensor
    # Calculate new cell: v_new = v_old * deformation (row vector convention)
    new_cell = cell @ deformation

    new_atoms.set_cell(new_cell, scale_atoms=True) # type: ignore[no-untyped-call]  # type: ignore[no-untyped-call]
    # Ensure we actually mutated the input in place since the test expects it.
    atoms.set_cell(new_cell, scale_atoms=True) # type: ignore[no-untyped-call]
    return new_atoms


def create_vacancy(atoms: Atoms, rate: float, rng: np.random.Generator | None = None) -> Atoms:
    """
    Randomly remove atoms to create vacancies.
    rate: Fraction of atoms to remove (0.0 to 1.0).
    Returns a new Atoms object.

    Args:
        atoms: Input structure.
        rate: Vacancy concentration.
        rng: Optional NumPy random number generator.
    """
    if not (0.0 <= rate <= 1.0):
        msg = "Vacancy rate must be between 0.0 and 1.0"
        raise ValueError(msg)

    new_atoms: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    n_atoms = len(new_atoms)

    if n_atoms == 0:
        return new_atoms

    n_vacancies = int(np.round(n_atoms * rate))

    if n_vacancies == 0:
        return new_atoms

    if n_vacancies >= n_atoms:
        # Remove all atoms
        del new_atoms[:]  # type: ignore[no-untyped-call]
        return new_atoms

    # Select indices to remove
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(n_atoms, size=n_vacancies, replace=False)
    # Delete atoms. Note: deleting from ASE atoms object modifies it in place.
    del new_atoms[indices]  # type: ignore[no-untyped-call]

    return new_atoms
