from pathlib import Path

from pyacemaker.core.report import ReportGenerator
from pyacemaker.domain_models.validation import (
    ElasticResult,
    PhononResult,
    ValidationReport,
    ValidationStatus,
)


def test_report_generation(tmp_path: Path) -> None:
    # Create dummy plot
    plot_path = tmp_path / "plot.png"
    plot_path.write_text("dummy image data")

    phonon = PhononResult(
        has_imaginary_modes=False, band_structure_path=plot_path, status=ValidationStatus.PASS
    )

    elastic = ElasticResult(
        c_ij={"C11": 100.0, "C12": 50.0},
        bulk_modulus=66.7,
        is_mechanically_stable=True,
        status=ValidationStatus.PASS,
    )

    report = ValidationReport(phonon=phonon, elastic=elastic, overall_status=ValidationStatus.PASS)

    generator = ReportGenerator()
    output_path = tmp_path / "report.html"

    generator.generate(report, output_path)

    assert output_path.exists()
    content = output_path.read_text()

    assert "Overall Status" in content
    assert "PASS" in content
    assert "Phonon Stability" in content
    assert "Elastic Stability" in content
    assert "100.00" in content

    import base64

    expected_b64 = base64.b64encode(b"dummy image data").decode("utf-8")
    assert expected_b64 in content
