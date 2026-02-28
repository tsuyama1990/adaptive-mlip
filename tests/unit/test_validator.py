from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.validator import LammpsInputValidator, Validator
from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult


class TestValidator:
    @pytest.fixture
    def mock_phonon_calc(self):
        return MagicMock()

    @pytest.fixture
    def mock_elastic_calc(self):
        return MagicMock()

    @pytest.fixture
    def mock_report_gen(self):
        return MagicMock()

    @pytest.fixture
    def validator(self, mock_phonon_calc, mock_elastic_calc, mock_report_gen):
        config = ValidationConfig()
        # Assuming Validator takes instances of calculators and report generator
        return Validator(
            config=config,
            phonon_calculator=mock_phonon_calc,
            elastic_calculator=mock_elastic_calc,
            report_generator=mock_report_gen,
        )

    def test_validate_pass(self, validator, mock_phonon_calc, mock_elastic_calc, mock_report_gen):
        mock_phonon_calc.check_stability.return_value = (True, "base64_phonon")
        mock_elastic_calc.calculate_properties.return_value = (
            True,
            {"C11": 100.0},
            150.0,
            "base64_elastic",
        )

        potential_path = Path("pot.yace")
        output_path = Path("report.html")
        structure = MagicMock()
        structure.get_volume.return_value = 10.0

        # Mock _relax_structure to isolate
        with patch.object(validator, "_relax_structure") as mock_relax:
            mock_relax.return_value = structure
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

    def test_validate_fail_phonon(self, validator, mock_phonon_calc, mock_elastic_calc):
        mock_phonon_calc.check_stability.return_value = (False, "base64_phonon_unstable")
        mock_elastic_calc.calculate_properties.return_value = (
            True,
            {"C11": 100.0},
            150.0,
            "base64_elastic",
        )

        potential_path = Path("pot.yace")
        output_path = Path("report.html")
        structure = MagicMock()
        structure.get_volume.return_value = 10.0

        with patch.object(validator, "_relax_structure") as mock_relax:
            mock_relax.return_value = structure
            result = validator.validate(potential_path, output_path, structure=structure)

        assert result.phonon_stable is False
        assert result.elastic_stable is True

    def test_relax_structure(self, validator, mock_elastic_calc):
        structure = MagicMock()
        pot_path = Path("pot.yace")

        # mock_elastic_calc.engine is accessed in _relax_structure
        mock_engine = MagicMock()
        mock_elastic_calc.engine = mock_engine
        mock_engine.relax.return_value = "relaxed_structure"

        relaxed = validator._relax_structure(structure, pot_path)

        assert relaxed == "relaxed_structure"
        mock_engine.relax.assert_called_once_with(structure, pot_path)

    def test_validate_structure_invalid_element(self):
        """Test rejection of structure with invalid chemical symbol (dummy X)."""
        # 'X' is in atomic_numbers but Z=0
        # Need pbc and cell for get_volume() check to pass first if we want to hit the element check.
        # Or let volume check fail? But volume check raises "Failed to compute structure volume"
        # We want to test element check specifically.
        # So we provide a valid cell.
        structure = Atoms("X", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        with pytest.raises(ValueError, match="dummy element"):
            LammpsInputValidator.validate_structure(structure)
