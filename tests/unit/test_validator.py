from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.validator import LammpsInputValidator, Validator
from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult


def test_lammps_input_validator_structure() -> None:
    with patch("pyacemaker.utils.validation.validate_structure") as mock_val:
        atoms = Atoms("H")
        LammpsInputValidator.validate_structure(atoms)
        mock_val.assert_called_once_with(atoms)


def test_lammps_input_validator_potential(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Safely mock the default directory to the temporary path
    pot_dir = tmp_path / "potentials"
    monkeypatch.setattr("pyacemaker.domain_models.defaults.DEFAULT_POTENTIALS_DIR", str(pot_dir))

    pot_dir.mkdir(parents=True, exist_ok=True)
    pot_path = pot_dir / "pot.yace"
    pot_path.touch()

    res = LammpsInputValidator.validate_potential(str(pot_path))
    assert res.resolve() == pot_path.resolve()

    # Clean up
    pot_path.unlink()


def test_lammps_input_validator_potential_none() -> None:
    with pytest.raises(ValueError, match="Validator requires a potential"):
        LammpsInputValidator.validate_potential(None)


def test_lammps_input_validator_potential_not_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pot_dir = tmp_path / "potentials"
    monkeypatch.setattr("pyacemaker.domain_models.defaults.DEFAULT_POTENTIALS_DIR", str(pot_dir))
    pot_dir.mkdir(parents=True, exist_ok=True)
    pot_path = pot_dir / "nonexistent.yace"

    with pytest.raises(FileNotFoundError, match="Potential file not found"):
        LammpsInputValidator.validate_potential(str(pot_path))


def test_lammps_input_validator_potential_not_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pot_dir = tmp_path / "potentials"
    monkeypatch.setattr("pyacemaker.domain_models.defaults.DEFAULT_POTENTIALS_DIR", str(pot_dir))
    pot_dir.mkdir(parents=True, exist_ok=True)
    pot_sub_dir = pot_dir / "pot_dir"
    pot_sub_dir.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Potential path is not a file"):
        LammpsInputValidator.validate_potential(str(pot_sub_dir))

    # Clean up
    pot_sub_dir.rmdir()


TEST_POTENTIAL_FILENAME = "pot.yace"
TEST_CELL = [10, 10, 10]
TEST_ELEM = "H"

class TestValidator:
    @pytest.fixture
    def mock_phonon_calc(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_elastic_calc(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_report_gen(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def validator(self, mock_phonon_calc: MagicMock, mock_elastic_calc: MagicMock, mock_report_gen: MagicMock) -> Validator:
        config = ValidationConfig()
        # Assuming Validator takes instances of calculators and report generator
        return Validator(
            config=config,
            phonon_calculator=mock_phonon_calc,
            elastic_calculator=mock_elastic_calc,
            report_generator=mock_report_gen,
        )

    def test_validate_pass(self, validator: Validator, mock_phonon_calc: MagicMock, mock_elastic_calc: MagicMock, mock_report_gen: MagicMock) -> None:
        mock_phonon_calc.check_stability.return_value = (True, "base64_phonon")
        mock_elastic_calc.calculate_properties.return_value = (
            True,
            {"C11": 100.0},
            150.0,
            "base64_elastic",
        )

        potential_path = Path(TEST_POTENTIAL_FILENAME)
        output_path = Path("report.html")
        structure = Atoms(TEST_ELEM, cell=TEST_CELL, pbc=True)

        # Configure elastic calc engine to return structure properly for _relax_structure
        mock_engine = MagicMock()
        mock_elastic_calc.engine = mock_engine
        mock_engine.relax.return_value = structure

        result = validator.validate(potential_path, output_path, structure=structure)

        assert isinstance(result, ValidationResult)
        assert result.phonon_stable is True
        assert result.elastic_stable is True
        assert result.c_ij["C11"] == 100.0
        assert result.bulk_modulus == 150.0
        assert result.plots["phonon"] == "base64_phonon"
        assert str(result.report_path) == str(output_path)

        mock_report_gen.generate.assert_called_once()
        mock_report_gen.save.assert_called_once()

    def test_validate_fail_phonon(self, validator: Validator, mock_phonon_calc: MagicMock, mock_elastic_calc: MagicMock) -> None:
        mock_phonon_calc.check_stability.return_value = (False, "base64_phonon_unstable")
        mock_elastic_calc.calculate_properties.return_value = (
            True,
            {"C11": 100.0},
            150.0,
            "base64_elastic",
        )

        potential_path = Path("pot.yace")
        output_path = Path("report.html")
        structure = Atoms("H", cell=[10, 10, 10], pbc=True)

        mock_engine = MagicMock()
        mock_elastic_calc.engine = mock_engine
        mock_engine.relax.return_value = structure

        result = validator.validate(potential_path, output_path, structure=structure)

        assert result.phonon_stable is False
        assert result.elastic_stable is True

    def test_relax_structure(self, validator: Validator, mock_elastic_calc: MagicMock) -> None:
        structure = MagicMock()
        pot_path = Path(TEST_POTENTIAL_FILENAME)

        # mock_elastic_calc.engine is accessed in _relax_structure
        mock_engine = MagicMock()
        mock_elastic_calc.engine = mock_engine
        mock_engine.relax.return_value = "relaxed_structure"

        relaxed = validator._relax_structure(structure, pot_path)

        assert relaxed == "relaxed_structure"
        mock_engine.relax.assert_called_once_with(structure, pot_path)

    def test_validate_structure_invalid_element(self) -> None:
        """Test rejection of structure with invalid chemical symbol (dummy X)."""
        structure = Atoms("X", positions=[[0, 0, 0]], cell=TEST_CELL, pbc=True)
        with pytest.raises(ValueError, match="dummy element"):
            LammpsInputValidator.validate_structure(structure)

        # Additional test cases to completely verify validation logic
        empty_structure = Atoms()
        with pytest.raises(ValueError, match="Structure is empty"):
            LammpsInputValidator.validate_structure(empty_structure)

        invalid_symbol_structure = Atoms(numbers=[0], positions=[[0, 0, 0]], cell=TEST_CELL, pbc=True)
        with pytest.raises(ValueError, match="dummy element"):
            LammpsInputValidator.validate_structure(invalid_symbol_structure)

    def test_validator_no_structure(self, validator: Validator) -> None:
        pot_path = Path(TEST_POTENTIAL_FILENAME)
        out_path = Path("report.html")

        with pytest.raises(ValueError, match="Validator requires a structure"):
            validator.validate(potential_path=pot_path, output_path=out_path, structure=None)
