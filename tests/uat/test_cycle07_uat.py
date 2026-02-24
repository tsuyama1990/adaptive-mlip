from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.validator import Validator
from pyacemaker.domain_models.validation import ValidationConfig


@pytest.fixture
def mock_config() -> ValidationConfig:
    return ValidationConfig(
        phonon_supercell=[2, 2, 2],
        elastic_strain=0.01,
        imaginary_frequency_tolerance=-0.05
    )

@pytest.fixture
def mock_atoms() -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]], cell=[5, 5, 5], pbc=True)

@pytest.fixture
def mock_potential_path(tmp_path: Path) -> Path:
    return tmp_path / "potential.yace"

def test_scenario_07_01_validate_potential(mock_config: ValidationConfig, mock_potential_path: Path, mock_atoms: Atoms) -> None:
    """
    Scenario 07-01: Validate Potential
    Verify that the system can run a full validation suite on a potential.
    """
    validator = Validator(mock_potential_path, mock_config, mock_atoms)

    # Mock everything to simulate a successful validation run
    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = mock_atoms.copy()  # type: ignore[no-untyped-call]

        with patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon, \
             patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic, \
             patch("pyacemaker.core.validator.ReportGenerator") as MockReport, \
             patch("pyacemaker.core.validator.LammpsDriver"):

            phonon_instance = MockPhonon.return_value
            phonon_instance.check_stability.return_value = (True, [])
            phonon_instance.get_band_structure_plot.return_value = "band_b64"
            phonon_instance.get_dos_plot.return_value = "dos_b64"

            elastic_instance = MockElastic.return_value
            elastic_instance.calculate.return_value = ([[1.0]], 100.0, 50.0)
            MockElastic.check_stability.return_value = True

            report_instance = MockReport.return_value
            report_instance.generate.return_value = "<html>PASS</html>"

            # Inject dependency
            validator.report_generator = report_instance

            result = validator.validate()

            assert result.phonon_stable is True
            assert result.elastic_stable is True
            assert result.imaginary_frequencies == []

            # Verify report generation called
            report_instance.save.assert_called_once()

def test_scenario_07_02_unstable_detection(mock_config: ValidationConfig, mock_potential_path: Path, mock_atoms: Atoms) -> None:
    """
    Scenario 07-02: Unstable Detection
    Verify that the system flags an unstable potential.
    """
    validator = Validator(mock_potential_path, mock_config, mock_atoms)

    # Mock failure in phonon stability
    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = mock_atoms.copy()  # type: ignore[no-untyped-call]

        with patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon, \
             patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic, \
             patch("pyacemaker.core.validator.ReportGenerator") as MockReport, \
             patch("pyacemaker.core.validator.LammpsDriver"):

            phonon_instance = MockPhonon.return_value
            phonon_instance.check_stability.return_value = (False, [-1.0, -0.5]) # Unstable!
            phonon_instance.get_band_structure_plot.return_value = "band_b64"
            phonon_instance.get_dos_plot.return_value = "dos_b64"

            elastic_instance = MockElastic.return_value
            elastic_instance.calculate.return_value = ([[1.0]], 100.0, 50.0)
            MockElastic.check_stability.return_value = True

            report_instance = MockReport.return_value
            report_instance.generate.return_value = "<html>FAIL</html>"

            # Inject dependency
            validator.report_generator = report_instance

            result = validator.validate()

            assert result.phonon_stable is False
            assert len(result.imaginary_frequencies) > 0

            # Verify report generation called
            report_instance.save.assert_called_once()
