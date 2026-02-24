import base64
from pathlib import Path

from jinja2 import Template

from pyacemaker.domain_models.validation import (
    ValidationReport,
)

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .error { color: orange; font-weight: bold; }
        .skipped { color: gray; font-weight: bold; }
        .section { margin-bottom: 20px; border: 1px solid #ccc; padding: 10px; }
        h2 { border-bottom: 1px solid #eee; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Validation Report</h1>
    <p>Overall Status: <span class="{{ report.overall_status.value.lower() }}">{{ report.overall_status.value }}</span></p>

    {% if report.phonon %}
    <div class="section">
        <h2>Phonon Stability</h2>
        <p>Status: <span class="{{ report.phonon.status.value.lower() }}">{{ report.phonon.status.value }}</span></p>
        <p>Imaginary Modes: {{ report.phonon.has_imaginary_modes }}</p>
        {% if phonon_plot_b64 %}
        <h3>Band Structure</h3>
        <img src="data:image/png;base64,{{ phonon_plot_b64 }}" alt="Phonon Band Structure" style="max-width: 100%;">
        {% endif %}
    </div>
    {% endif %}

    {% if report.elastic %}
    <div class="section">
        <h2>Elastic Stability</h2>
        <p>Status: <span class="{{ report.elastic.status.value.lower() }}">{{ report.elastic.status.value }}</span></p>
        <p>Mechanically Stable: {{ report.elastic.is_mechanically_stable }}</p>
        <p>Bulk Modulus: {{ "%.2f"|format(report.elastic.bulk_modulus) }} GPa</p>
        <h3>Elastic Constants (GPa)</h3>
        <table>
            <tr><th>Component</th><th>Value</th></tr>
            {% for key, value in report.elastic.c_ij.items() %}
            <tr><td>{{ key }}</td><td>{{ "%.2f"|format(value) }}</td></tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
</body>
</html>
"""


class ReportGenerator:
    """Generates HTML validation report."""

    def generate(self, report: ValidationReport, output_path: Path) -> None:
        """Generates the HTML report."""
        phonon_plot_b64 = ""
        if (
            report.phonon
            and report.phonon.band_structure_path
            and report.phonon.band_structure_path.exists()
        ):
            with report.phonon.band_structure_path.open("rb") as f:
                phonon_plot_b64 = base64.b64encode(f.read()).decode("utf-8")

        template = Template(REPORT_TEMPLATE)
        html_content = template.render(report=report, phonon_plot_b64=phonon_plot_b64)

        with output_path.open("w") as f:
            f.write(html_content)
