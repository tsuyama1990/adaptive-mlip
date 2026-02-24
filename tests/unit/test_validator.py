from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.validator import Validator
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
            report_generator=mock_report_gen
        )

    def test_validate_pass(self, validator, mock_phonon_calc, mock_elastic_calc, mock_report_gen):
        mock_phonon_calc.check_stability.return_value = (True, "base64_phonon")
        mock_elastic_calc.calculate_properties.return_value = (True, {"C11": 100.0}, 150.0, "base64_elastic")

        potential_path = Path("pot.yace")
        output_path = Path("report.html")
        structure = MagicMock()

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
        mock_elastic_calc.calculate_properties.return_value = (True, {"C11": 100.0}, 150.0, "base64_elastic")

        potential_path = Path("pot.yace")
        output_path = Path("report.html")
        structure = MagicMock()

        with patch.object(validator, "_relax_structure") as mock_relax:
            mock_relax.return_value = structure
            result = validator.validate(potential_path, output_path, structure=structure)

        assert result.phonon_stable is False
        assert result.elastic_stable is True
