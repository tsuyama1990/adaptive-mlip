from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.validator import Validator
from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult


@pytest.fixture
def mock_config() -> ValidationConfig:
    return ValidationConfig(
        phonon_supercell=[2, 2, 2],
        elastic_strain=0.01
    )

@pytest.fixture
def mock_atoms() -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]], cell=[5, 5, 5], pbc=True)

@pytest.fixture
def mock_potential_path(tmp_path: Path) -> Path:
    return tmp_path / "potential.yace"

@pytest.fixture
def mock_lammps_driver() -> Generator[MagicMock, None, None]:
    with patch("pyacemaker.core.validator.LammpsDriver") as mock:
        yield mock

def test_validator_init(mock_config: ValidationConfig, mock_potential_path: Path, mock_atoms: Atoms, mock_lammps_driver: MagicMock) -> None:
    validator = Validator(mock_potential_path, mock_config, mock_atoms)
    assert validator.potential_path == mock_potential_path
    assert validator.config == mock_config
    assert validator.structure == mock_atoms
    mock_lammps_driver.assert_called_once()

def test_validate_success(mock_config: ValidationConfig, mock_potential_path: Path, mock_atoms: Atoms, mock_lammps_driver: MagicMock) -> None:
    validator = Validator(mock_potential_path, mock_config, mock_atoms)

    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = mock_atoms.copy()  # type: ignore[no-untyped-call]

        with patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon, \
             patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic, \
             patch("pyacemaker.core.validator.ReportGenerator") as MockReport:

            phonon_instance = MockPhonon.return_value
            phonon_instance.check_stability.return_value = (True, [])
            phonon_instance.get_band_structure_plot.return_value = "band_b64"
            phonon_instance.get_dos_plot.return_value = "dos_b64"

            elastic_instance = MockElastic.return_value
            elastic_instance.calculate.return_value = ([[1.0]], 100.0, 50.0)
            MockElastic.check_stability.return_value = True

            report_instance = MockReport.return_value
            report_instance.generate.return_value = "<html></html>"

            result = validator.validate()

            assert isinstance(result, ValidationResult)
            assert result.phonon_stable is True
            assert result.elastic_stable is True
            assert result.plots
            assert result.plots == {"band_structure": "band_b64", "dos": "dos_b64"}

def test_validate_failure_phonon(mock_config: ValidationConfig, mock_potential_path: Path, mock_atoms: Atoms, mock_lammps_driver: MagicMock) -> None:
    validator = Validator(mock_potential_path, mock_config, mock_atoms)

    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = mock_atoms.copy()  # type: ignore[no-untyped-call]

        with patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon, \
             patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic, \
             patch("pyacemaker.core.validator.ReportGenerator"):

            phonon_instance = MockPhonon.return_value
            phonon_instance.check_stability.return_value = (False, [-1.0])
            # Must return strings for plots even if failing stability, or raise exception
            phonon_instance.get_band_structure_plot.side_effect = Exception("Plot failed")
            phonon_instance.get_dos_plot.side_effect = Exception("Plot failed")

            elastic_instance = MockElastic.return_value
            elastic_instance.calculate.return_value = ([[1.0]], 100.0, 50.0)
            MockElastic.check_stability.return_value = True

            result = validator.validate()
            assert result.phonon_stable is False
            assert result.imaginary_frequencies == [-1.0]
            assert result.plots == {}

def test_validate_failure_elastic(mock_config: ValidationConfig, mock_potential_path: Path, mock_atoms: Atoms, mock_lammps_driver: MagicMock) -> None:
    validator = Validator(mock_potential_path, mock_config, mock_atoms)

    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = mock_atoms.copy()  # type: ignore[no-untyped-call]

        with patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon, \
             patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic, \
             patch("pyacemaker.core.validator.ReportGenerator"):

            phonon_instance = MockPhonon.return_value
            phonon_instance.check_stability.return_value = (True, [])
            phonon_instance.get_band_structure_plot.return_value = "band_b64"
            phonon_instance.get_dos_plot.return_value = "dos_b64"

            elastic_instance = MockElastic.return_value
            elastic_instance.calculate.return_value = ([[-1.0]], -10.0, 5.0)
            MockElastic.check_stability.return_value = False

            result = validator.validate()
            assert result.elastic_stable is False
            assert result.bulk_modulus == -10.0
