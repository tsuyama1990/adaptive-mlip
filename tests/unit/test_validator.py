from pathlib import Path
from typing import Any

import pytest
from ase import Atoms

from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import MDConfig, PyAceConfig, ValidationConfig
from pyacemaker.domain_models.validation import ElasticResult, PhononResult, ValidationStatus


@pytest.fixture
def mock_config(mocker: Any) -> Any:
    config = mocker.Mock(spec=PyAceConfig)
    config.validation = ValidationConfig()
    config.md = MDConfig(
        thermo_freq=10,
        dump_freq=10,
        n_steps=100,
        timestep=0.001,
        temperature=300.0,
        pressure=1.0,
    )
    return config


def test_validator_validate(mocker: Any, mock_config: PyAceConfig, tmp_path: Path) -> None:
    validator = Validator(mock_config)

    mock_relax = mocker.patch("pyacemaker.core.validator.relax_structure")
    relaxed_atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    mock_relax.return_value = relaxed_atoms

    mock_phonon_cls = mocker.patch("pyacemaker.core.validator.PhononCalculator")
    mock_phonon = mock_phonon_cls.return_value
    mock_phonon.calculate.return_value = PhononResult(
        has_imaginary_modes=False,
        status=ValidationStatus.PASS,
        band_structure_path=tmp_path / "phonon.png",
    )

    mock_elastic_cls = mocker.patch("pyacemaker.core.validator.ElasticCalculator")
    mock_elastic = mock_elastic_cls.return_value
    mock_elastic.calculate.return_value = ElasticResult(
        c_ij={}, bulk_modulus=100.0, is_mechanically_stable=True, status=ValidationStatus.PASS
    )

    mock_report_gen = mocker.Mock()
    validator.report_generator = mock_report_gen

    base_structure = Atoms("H", cell=[10, 10, 10], pbc=True)
    potential_path = Path("dummy.yace")
    output_dir = tmp_path / "validation"

    report = validator.validate(potential_path, base_structure, output_dir)

    assert report.overall_status == ValidationStatus.PASS
    assert report.phonon
    assert report.phonon.status == ValidationStatus.PASS
    assert report.elastic
    assert report.elastic.status == ValidationStatus.PASS

    mock_relax.assert_called_once()
    mock_phonon.calculate.assert_called_once()
    mock_elastic.calculate.assert_called_once()
    mock_report_gen.generate.assert_called_once()


def test_validator_validate_fail(mocker: Any, mock_config: PyAceConfig, tmp_path: Path) -> None:
    validator = Validator(mock_config)

    mock_relax = mocker.patch("pyacemaker.core.validator.relax_structure")
    relaxed_atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    mock_relax.return_value = relaxed_atoms

    mock_phonon_cls = mocker.patch("pyacemaker.core.validator.PhononCalculator")
    mock_phonon = mock_phonon_cls.return_value
    # Fail phonon
    mock_phonon.calculate.return_value = PhononResult(
        has_imaginary_modes=True,
        status=ValidationStatus.FAIL,
        band_structure_path=tmp_path / "phonon.png",
    )

    mock_elastic_cls = mocker.patch("pyacemaker.core.validator.ElasticCalculator")
    mock_elastic = mock_elastic_cls.return_value
    mock_elastic.calculate.return_value = ElasticResult(
        c_ij={}, bulk_modulus=100.0, is_mechanically_stable=True, status=ValidationStatus.PASS
    )

    mock_report_gen = mocker.Mock()
    validator.report_generator = mock_report_gen

    base_structure = Atoms("H", cell=[10, 10, 10], pbc=True)
    potential_path = Path("dummy.yace")
    output_dir = tmp_path / "validation"

    report = validator.validate(potential_path, base_structure, output_dir)

    assert report.overall_status == ValidationStatus.FAIL
