from pathlib import Path

from jinja2 import Environment, select_autoescape

from pyacemaker.domain_models.validation import ValidationResult


class ReportGenerator:
    """
    Generates HTML validation reports using Jinja2 templates.
    """

    TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Potential Validation Report</title>
        <style>
            body { font-family: sans-serif; margin: 2rem; }
            h1 { color: #333; }
            .section { margin-bottom: 2rem; border: 1px solid #ddd; padding: 1rem; border-radius: 5px; }
            .status { font-weight: bold; }
            .pass { color: green; }
            .fail { color: red; }
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Validation Report</h1>

        <div class="section">
            <h2>Summary</h2>
            <p><strong>Dynamical Stability (Phonons):</strong>
                <span class="status {{ 'pass' if result.phonon_stable else 'fail' }}">
                    {{ 'Stable' if result.phonon_stable else 'Unstable' }}
                </span>
            </p>
            <p><strong>Mechanical Stability (Elastic):</strong>
                <span class="status {{ 'pass' if result.elastic_stable else 'fail' }}">
                    {{ 'Stable' if result.elastic_stable else 'Unstable' }}
                </span>
            </p>
        </div>

        <div class="section">
            <h2>Elastic Properties</h2>
            <table>
                <tr><th>Property</th><th>Value (GPa)</th></tr>
                {% for key, value in result.c_ij.items() %}
                <tr><td>{{ key }}</td><td>{{ "%.2f"|format(value) }}</td></tr>
                {% endfor %}
                <tr><td>Bulk Modulus</td><td>{{ "%.2f"|format(result.bulk_modulus) }}</td></tr>
            </table>
            {% if result.plots.elastic %}
            <h3>Stress-Strain Curves</h3>
            <img src="data:image/png;base64,{{ result.plots.elastic }}" alt="Stress-Strain Plot">
            {% endif %}
        </div>

        <div class="section">
            <h2>Phonon Band Structure</h2>
            {% if result.plots.phonon %}
            <img src="data:image/png;base64,{{ result.plots.phonon }}" alt="Phonon Band Structure">
            {% else %}
            <p>No plot available.</p>
            {% endif %}
        </div>
    </body>
    </html>
    """

    def __init__(self) -> None:
        self.env = Environment(autoescape=select_autoescape(["html", "xml"]))
        self.template = self.env.from_string(self.TEMPLATE)

    def generate(self, result: ValidationResult) -> str:
        """Generates the HTML content."""
        return self.template.render(result=result)

    def save(self, path: Path, content: str) -> None:
        """Saves the report to a file."""
        path.write_text(content, encoding="utf-8")
