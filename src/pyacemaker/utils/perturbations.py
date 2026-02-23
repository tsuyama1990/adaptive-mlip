import numpy as np
from ase import Atoms


def rattle(atoms: Atoms, stdev: float) -> Atoms:
    """
    Apply random Gaussian noise to atomic positions.
    Returns a new Atoms object.
    """
    if stdev < 0:
        msg = "Standard deviation must be non-negative"
        raise ValueError(msg)

    new_atoms: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    if stdev > 0:
        # Use numpy directly to ensure randomness and avoid ASE version ambiguities
        rng = np.random.default_rng()
        displacement = rng.normal(scale=stdev, size=new_atoms.positions.shape)
        new_atoms.positions += displacement
    return new_atoms


def apply_strain(atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
    """
    Apply strain tensor to the unit cell.
    strain_tensor: 3x3 numpy array representing the strain epsilon.
    New cell = Old cell @ (I + epsilon)
    Atoms are scaled accordingly.
    """
    if strain_tensor.shape != (3, 3):
        msg = "Strain tensor must be a 3x3 matrix"
        raise ValueError(msg)

    new_atoms: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    cell = new_atoms.get_cell()  # type: ignore[no-untyped-call]

    deformation = np.eye(3) + strain_tensor
    # Calculate new cell: v_new = v_old * deformation (row vector convention)
    new_cell = cell @ deformation

    new_atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]
    return new_atoms


def create_vacancy(atoms: Atoms, rate: float) -> Atoms:
    """
    Randomly remove atoms to create vacancies.
    rate: Fraction of atoms to remove (0.0 to 1.0).
    Returns a new Atoms object.
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
    indices = np.random.choice(n_atoms, size=n_vacancies, replace=False)
    # Delete atoms. Note: deleting from ASE atoms object modifies it in place.
    del new_atoms[indices]  # type: ignore[no-untyped-call]

    return new_atoms
