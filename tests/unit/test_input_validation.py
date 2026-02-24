from pathlib import Path

import pytest
from ase import Atoms

from pyacemaker.core.input_validation import LammpsValidator


def test_validate_structure_valid() -> None:
    structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    # Should not raise
    LammpsValidator.validate_structure(structure)


def test_validate_structure_invalid_none() -> None:
    with pytest.raises(ValueError, match="Structure must be provided"):
        LammpsValidator.validate_structure(None)


def test_validate_structure_invalid_type() -> None:
    with pytest.raises(TypeError, match="Expected ASE Atoms object"):
        LammpsValidator.validate_structure("not an atoms object")


def test_validate_structure_invalid_empty() -> None:
    with pytest.raises(ValueError, match="Structure contains no atoms"):
        LammpsValidator.validate_structure(Atoms())


def test_validate_potential_valid(tmp_path: Path) -> None:
    pot_path = tmp_path / "test.pot"
    pot_path.touch()
    validated = LammpsValidator.validate_potential(pot_path)
    assert validated == pot_path


def test_validate_potential_invalid_none() -> None:
    with pytest.raises(ValueError, match="Potential path must be provided"):
        LammpsValidator.validate_potential(None)


def test_validate_potential_invalid_missing() -> None:
    with pytest.raises(FileNotFoundError, match="Potential file not found"):
        LammpsValidator.validate_potential("non_existent.pot")
