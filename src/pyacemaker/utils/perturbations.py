from typing import Literal

import numpy as np
from ase import Atoms


def rattle(atoms: Atoms, stdev: float) -> Atoms:
    """
    Applies random Gaussian noise to atomic positions.

    Args:
        atoms: The structure to perturb.
        stdev: Standard deviation of the Gaussian noise (in Angstroms).

    Returns:
        A new Atoms object with perturbed positions.
    """
    new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
    noise = np.random.normal(0.0, stdev, size=new_atoms.positions.shape)
    new_atoms.positions += noise
    return new_atoms  # type: ignore[no-any-return]


def apply_strain(
    atoms: Atoms, strain_range: float, mode: Literal["volume", "shear", "full"] = "full"
) -> Atoms:
    """
    Applies random strain tensor to the unit cell.

    Args:
        atoms: The structure to strain.
        strain_range: Maximum magnitude of strain components (e.g. 0.05 for 5%).
        mode: Type of strain to apply ("volume", "shear", "full").

    Returns:
        A new Atoms object with distorted cell and scaled positions.
    """
    new_atoms = atoms.copy()  # type: ignore[no-untyped-call]

    strain = np.zeros((3, 3))

    if mode in ("volume", "full"):
        # Diagonal elements (normal strain)
        diag = np.random.uniform(-strain_range, strain_range, size=3)
        np.fill_diagonal(strain, diag)

    if mode in ("shear", "full"):
        # Off-diagonal elements (shear strain)
        off_diag = np.random.uniform(-strain_range, strain_range, size=3)
        strain[0, 1] = strain[1, 0] = off_diag[0]
        strain[0, 2] = strain[2, 0] = off_diag[1]
        strain[1, 2] = strain[2, 1] = off_diag[2]

    # Deformation gradient F = I + epsilon (small strain approximation)
    deformation_gradient = np.eye(3) + strain

    # Apply deformation to cell vectors
    # ASE cell is row-major: [v1, v2, v3]
    original_cell = new_atoms.get_cell()
    new_cell = original_cell @ deformation_gradient

    new_atoms.set_cell(new_cell, scale_atoms=True)
    return new_atoms  # type: ignore[no-any-return]


def create_vacancy(atoms: Atoms, vacancy_rate: float) -> Atoms:
    """
    Creates vacancies by removing random atoms.

    Args:
        atoms: The structure to modify.
        vacancy_rate: Fraction of atoms to remove (0.0 to 1.0).

    Returns:
        A new Atoms object with vacancies.
    """
    new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
    n_atoms = len(new_atoms)

    if n_atoms <= 1:
        return new_atoms  # type: ignore[no-any-return]

    n_vacancies = int(np.round(n_atoms * vacancy_rate))

    # Ensure at least one vacancy if rate > 0 is requested, but check logic
    if n_vacancies == 0 and vacancy_rate > 0.01:  # Heuristic
        n_vacancies = 1

    if n_vacancies >= n_atoms:
        n_vacancies = n_atoms - 1

    if n_vacancies > 0:
        indices_to_remove = np.random.choice(range(n_atoms), size=n_vacancies, replace=False)
        del new_atoms[indices_to_remove]

    return new_atoms  # type: ignore[no-any-return]
