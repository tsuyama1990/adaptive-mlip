from unittest.mock import mock_open, patch

import pytest

from pyacemaker.core.report import ReportGenerator
from pyacemaker.domain_models.validation import ValidationResult


@pytest.fixture
def mock_result() -> ValidationResult:
    return ValidationResult(
        phonon_stable=True,
        elastic_stable=False,
        imaginary_frequencies=[],
        elastic_tensor=[[1.0, 0.0], [0.0, 1.0]],
        bulk_modulus=100.0,
        shear_modulus=50.0,
        plots={"band": "base64string"},
    )

def test_report_generator_init() -> None:
    generator = ReportGenerator()
    assert generator is not None

def test_generate_html(mock_result: ValidationResult) -> None:
    generator = ReportGenerator()

    with patch("builtins.open", mock_open(read_data="template")), \
         patch("pyacemaker.core.report.Template") as mock_template:

        mock_instance = mock_template.return_value
        mock_instance.render.return_value = "<html>Rendered</html>"

        html = generator.generate(mock_result)

        assert html == "<html>Rendered</html>"
        mock_instance.render.assert_called_with(result=mock_result)

def test_save_report(mock_result: ValidationResult) -> None:
    generator = ReportGenerator()

    with patch("pyacemaker.core.report.ReportGenerator.generate", return_value="<html></html>"), \
         patch("pathlib.Path.write_text") as mock_write:

        generator.save(mock_result, "report.html")
        mock_write.assert_called_with("<html></html>", encoding="utf-8")
