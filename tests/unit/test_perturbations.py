import numpy as np
from ase.build import bulk

from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle


def test_rattle_preserves_stoichiometry() -> None:
    atoms = bulk("Fe", cubic=True)
    n_atoms = len(atoms)
    rattled = rattle(atoms, stdev=0.1)

    assert len(rattled) == n_atoms
    assert rattled.get_chemical_formula() == atoms.get_chemical_formula()
    # Ensure positions changed
    assert not np.allclose(atoms.get_positions(), rattled.get_positions())
    # Ensure cell didn't change
    assert np.allclose(atoms.get_cell(), rattled.get_cell())


def test_apply_strain_changes_cell() -> None:
    atoms = bulk("Fe", cubic=True)
    original_cell = atoms.get_cell()

    # Volume strain
    strained_vol = apply_strain(atoms, strain_range=0.1, mode="volume")
    assert not np.allclose(strained_vol.get_cell(), original_cell)
    # Shear strain might not change volume much but changes cell vectors
    strained_shear = apply_strain(atoms, strain_range=0.1, mode="shear")
    assert not np.allclose(strained_shear.get_cell(), original_cell)


def test_create_vacancy_removes_atom() -> None:
    # 2x2x2 supercell of Fe (2 atoms per cell) -> 16 atoms
    atoms = bulk("Fe", cubic=True) * (2, 2, 2)
    n_atoms = len(atoms)

    # 0.1 rate -> 1.6 -> 1 or 2 atoms? Let's assume round or int.
    # Usually vacancy implies integer removal.
    # If I implement it as max(1, int(n*rate)) it ensures at least 1 vacancy if rate > 0?
    # Or strict int(n*rate).
    # Let's assume the implementation will handle it. I'll test that count decreases.

    vacant = create_vacancy(atoms, vacancy_rate=0.2)  # 3.2 atoms -> 3 atoms
    assert len(vacant) < n_atoms
    # It should be a copy
    assert vacant is not atoms
