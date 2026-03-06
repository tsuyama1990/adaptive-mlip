from pyacemaker.core.report import ReportGenerator
from pyacemaker.domain_models.validation import ValidationResult


def test_generate_report_pass(tmp_path):
    result = ValidationResult(
        phonon_stable=True,
        elastic_stable=True,
        c_ij={"C11": 200.0},
        bulk_modulus=150.0,
        plots={"phonon": "base64_string", "elastic": "base64_string"},
        report_path=str(tmp_path / "report.html"),
    )
    generator = ReportGenerator()
    html = generator.generate(result)
    assert "Validation Report" in html
    assert "Stable" in html
    assert "200.0" in html
    assert "base64_string" in html


def test_generate_report_fail(tmp_path):
    result = ValidationResult(
        phonon_stable=False,
        elastic_stable=True,
        c_ij={"C11": 200.0},
        bulk_modulus=150.0,
        plots={"phonon": "base64_string", "elastic": "base64_string"},
        report_path=str(tmp_path / "report.html"),
    )
    generator = ReportGenerator()
    html = generator.generate(result)
    assert "Validation Report" in html
    assert "Unstable" in html
