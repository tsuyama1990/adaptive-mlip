from pathlib import Path

import pytest
from ase import Atoms

from pyacemaker.core.validator import LammpsValidator


def test_validate_structure_valid() -> None:
    atoms = Atoms("H")
    LammpsValidator.validate_structure(atoms)


def test_validate_structure_none() -> None:
    with pytest.raises(ValueError, match="Structure must be provided"):
        LammpsValidator.validate_structure(None)


def test_validate_structure_invalid_type() -> None:
    with pytest.raises(TypeError, match="Expected ASE Atoms object"):
        LammpsValidator.validate_structure("not atoms")


def test_validate_structure_empty() -> None:
    atoms = Atoms()
    with pytest.raises(ValueError, match="Structure contains no atoms"):
        LammpsValidator.validate_structure(atoms)


def test_validate_potential_valid(tmp_path: Path) -> None:
    p = tmp_path / "pot.yace"
    p.touch()
    res = LammpsValidator.validate_potential(p)
    assert res == p


def test_validate_potential_missing() -> None:
    with pytest.raises(FileNotFoundError, match="Potential file not found"):
        LammpsValidator.validate_potential("missing.yace")


def test_validate_potential_none() -> None:
    with pytest.raises(ValueError, match="Potential path must be provided"):
        LammpsValidator.validate_potential(None)
