from pathlib import Path

from jinja2 import Template

from pyacemaker.domain_models.validation import ValidationResult

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Potential Validation Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .section { margin-bottom: 30px; }
        .success { color: green; font-weight: bold; }
        .failure { color: red; font-weight: bold; }
        .plot { max-width: 800px; border: 1px solid #ccc; margin: 10px 0; }
        table { border-collapse: collapse; width: auto; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        th { background-color: #f2f2f2; text-align: center; }
    </style>
</head>
<body>
    <h1>Validation Report</h1>

    <div class="section">
        <h2>Summary</h2>
        <p>Phonon Stability: <span class="{{ 'success' if result.phonon_stable else 'failure' }}">{{ 'PASS' if result.phonon_stable else 'FAIL' }}</span></p>
        <p>Elastic Stability: <span class="{{ 'success' if result.elastic_stable else 'failure' }}">{{ 'PASS' if result.elastic_stable else 'FAIL' }}</span></p>
    </div>

    <div class="section">
        <h2>Phonons</h2>
        {% if result.imaginary_frequencies %}
            <p class="failure">Imaginary Frequencies Detected (THz): {{ result.imaginary_frequencies }}</p>
        {% else %}
            <p class="success">No imaginary frequencies detected.</p>
        {% endif %}

        {% if result.plots.band_structure %}
            <h3>Band Structure</h3>
            <img class="plot" src="data:image/png;base64,{{ result.plots.band_structure }}" alt="Band Structure" />
        {% endif %}

        {% if result.plots.dos %}
            <h3>Density of States</h3>
            <img class="plot" src="data:image/png;base64,{{ result.plots.dos }}" alt="DOS" />
        {% endif %}
    </div>

    <div class="section">
        <h2>Elastic Properties</h2>
        <p><strong>Bulk Modulus (B):</strong> {{ "%.2f"|format(result.bulk_modulus) }} GPa</p>
        <p><strong>Shear Modulus (G):</strong> {{ "%.2f"|format(result.shear_modulus) }} GPa</p>

        <h3>Elastic Tensor (Cij) [GPa]</h3>
        <table>
            {% for row in result.elastic_tensor %}
            <tr>
                {% for val in row %}
                <td>{{ "%.2f"|format(val) }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

class ReportGenerator:
    """Generates HTML reports for validation results."""

    def generate(self, result: ValidationResult) -> str:
        """
        Generates HTML string from ValidationResult.
        """
        template = Template(TEMPLATE)
        return template.render(result=result)

    def save(self, result: ValidationResult, path: str | Path) -> None:
        """
        Generates and saves the report to a file.
        """
        html = self.generate(result)
        Path(path).write_text(html, encoding="utf-8")
