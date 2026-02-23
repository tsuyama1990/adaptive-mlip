import numpy as np
import pytest
from ase.build import bulk

from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle


def test_rattle() -> None:
    atoms = bulk("Al", "fcc", a=4.0)
    original_pos = atoms.get_positions()  # type: ignore[no-untyped-call]

    # Rattle
    new_atoms = rattle(atoms, stdev=0.1)

    # Check that positions changed
    assert not np.allclose(original_pos, new_atoms.get_positions())  # type: ignore[no-untyped-call]

    # Check that cell is same
    assert np.allclose(atoms.get_cell(), new_atoms.get_cell())  # type: ignore[no-untyped-call]

    # Check that original atoms not modified
    assert np.allclose(atoms.get_positions(), original_pos)  # type: ignore[no-untyped-call]

    # Check error
    with pytest.raises(ValueError, match="non-negative"):
        rattle(atoms, stdev=-0.1)


def test_apply_strain() -> None:
    atoms = bulk("Al", "fcc", a=4.0)
    original_cell = atoms.get_cell()  # type: ignore[no-untyped-call]

    # Apply hydrostatic strain (volume change)
    strain = np.eye(3) * 0.1  # 10% expansion
    new_atoms = apply_strain(atoms, strain)

    # New cell should be 1.1 * old_cell (since I + 0.1*I = 1.1*I)
    expected_cell = original_cell * 1.1
    assert np.allclose(new_atoms.get_cell(), expected_cell)  # type: ignore[no-untyped-call]

    # Check error
    with pytest.raises(ValueError, match="3x3 matrix"):
        apply_strain(atoms, np.zeros((2, 2)))


def test_create_vacancy() -> None:
    # Use a larger supercell
    atoms = bulk("Al", "fcc", a=4.0).repeat((3, 3, 3))  # 4 atoms * 27 = 108 atoms
    n_original = len(atoms)

    # Remove 10%
    new_atoms = create_vacancy(atoms, rate=0.1)
    n_expected = int(np.round(n_original * 0.1))

    assert len(new_atoms) == n_original - n_expected

    # Test rate 0
    assert len(create_vacancy(atoms, rate=0.0)) == n_original

    # Test rate 1
    assert len(create_vacancy(atoms, rate=1.0)) == 0

    # Test error
    with pytest.raises(ValueError, match="Vacancy rate"):
        create_vacancy(atoms, rate=1.5)
