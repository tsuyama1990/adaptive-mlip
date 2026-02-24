from pathlib import Path
from typing import ClassVar

from jinja2 import Template

from pyacemaker.domain_models.validation import ValidationResult


class ReportGenerator:
    """
    Generates HTML validation reports.
    """

    # Cache the template as a class variable to avoid reloading it on every instantiation
    _TEMPLATE: ClassVar[Template | None] = None

    _DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .section { margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; }
        .status { font-weight: bold; }
        .pass { color: green; }
        .fail { color: red; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Validation Report</h1>

    <div class="section">
        <h2>Summary</h2>
        <p>Phonon Stable: <span class="status {{ 'pass' if result.phonon_stable else 'fail' }}">{{ result.phonon_stable }}</span></p>
        <p>Elastic Stable: <span class="status {{ 'pass' if result.elastic_stable else 'fail' }}">{{ result.elastic_stable }}</span></p>
    </div>

    <div class="section">
        <h2>Elastic Properties</h2>
        <p>Bulk Modulus: {{ "%.2f"|format(result.bulk_modulus) }} GPa</p>
        <p>Shear Modulus: {{ "%.2f"|format(result.shear_modulus) }} GPa</p>
    </div>

    {% if not result.phonon_stable %}
    <div class="section">
        <h2>Instabilities</h2>
        <p>Imaginary Frequencies: {{ result.imaginary_frequencies }}</p>
    </div>
    {% endif %}

    <div class="section">
        <h2>Plots</h2>
        {% if result.plots.get('band_structure') %}
        <h3>Band Structure</h3>
        <img src="data:image/png;base64,{{ result.plots['band_structure'] }}" />
        {% endif %}

        {% if result.plots.get('dos') %}
        <h3>Density of States</h3>
        <img src="data:image/png;base64,{{ result.plots['dos'] }}" />
        {% endif %}
    </div>
</body>
</html>
"""

    def generate(self, result: ValidationResult) -> str:
        """
        Generates the HTML content.
        """
        template = self._get_template()
        return template.render(result=result)

    def save(self, result: ValidationResult, filepath: str | Path) -> None:
        """
        Generates and saves the report to a file.
        """
        html = self.generate(result)
        Path(filepath).write_text(html, encoding="utf-8")

    @classmethod
    def _get_template(cls) -> Template:
        """
        Returns the cached template or loads/creates it.
        """
        if cls._TEMPLATE is None:
            # In a real scenario, this might load from a file resource.
            # Here we use the inline string for simplicity and portability.
            cls._TEMPLATE = Template(cls._DEFAULT_TEMPLATE)
        return cls._TEMPLATE
