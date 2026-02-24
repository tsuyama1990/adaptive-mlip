import logging
from pathlib import Path

from ase import Atoms

from pyacemaker.core.report import ReportGenerator
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.defaults import DEFAULT_VALIDATION_REPORT_FILENAME
from pyacemaker.domain_models.validation import (
    ValidationReport,
    ValidationStatus,
)
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.lammps import relax_structure
from pyacemaker.utils.phonons import PhononCalculator

logger = logging.getLogger(__name__)


class Validator:
    """The Guardian: Validates potentials using physical checks."""

    def __init__(self, config: PyAceConfig) -> None:
        self.config = config.validation
        self.md_config = config.md
        self.report_generator = ReportGenerator()

    def validate(
        self, potential_path: Path, base_structure: Atoms, output_dir: Path
    ) -> ValidationReport:
        """
        Runs validation suite.
        1. Relax structure.
        2. Phonons.
        3. Elastic constants.
        4. Report.
        """
        if not self.config.enabled:
            return ValidationReport(overall_status=ValidationStatus.SKIPPED)

        logger.info(f"Starting validation for potential: {potential_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Relax Structure
        logger.info("Relaxing base structure...")
        try:
            relaxed_structure = relax_structure(base_structure, potential_path, self.md_config)
        except Exception:
            logger.exception("Structure relaxation failed")
            return ValidationReport(overall_status=ValidationStatus.ERROR)

        # 2. Phonons
        logger.info("Running Phonon calculation...")
        phonon_calc = PhononCalculator(self.config.phonon, self.md_config)
        try:
            phonon_result = phonon_calc.calculate(relaxed_structure, potential_path, output_dir)
        except Exception:
            logger.exception("Phonon calculation failed")
            phonon_result = None

        # 3. Elastic
        logger.info("Running Elastic calculation...")
        elastic_calc = ElasticCalculator(self.config.elastic, self.md_config)
        try:
            elastic_result = elastic_calc.calculate(relaxed_structure, potential_path)
        except Exception:
            logger.exception("Elastic calculation failed")
            elastic_result = None

        # Determine overall status
        status = ValidationStatus.PASS
        if phonon_result and phonon_result.status == ValidationStatus.FAIL:
            status = ValidationStatus.FAIL
        if elastic_result and elastic_result.status == ValidationStatus.FAIL:
            status = ValidationStatus.FAIL

        if phonon_result is None or elastic_result is None:
            status = ValidationStatus.ERROR

        report = ValidationReport(
            phonon=phonon_result,
            elastic=elastic_result,
            overall_status=status,
            report_path=output_dir / DEFAULT_VALIDATION_REPORT_FILENAME,
        )

        # 4. Generate Report
        try:
            if report.report_path:
                self.report_generator.generate(report, report.report_path)
                logger.info(f"Validation report generated: {report.report_path}")
        except Exception:
            logger.exception("Failed to generate report")

        return report
