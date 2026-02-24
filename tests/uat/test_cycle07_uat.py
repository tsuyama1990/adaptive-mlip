from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.defaults import DEFAULT_VALIDATION_REPORT_FILENAME
from pyacemaker.domain_models.validation import (
    ElasticResult,
    PhononResult,
    ValidationStatus,
)


@pytest.fixture
def uat_config(tmp_path: Path) -> PyAceConfig:
    (tmp_path / "Fe.UPF").touch()
    config_dict = {
        "project_name": "UAT_Project",
        "structure": {
            "elements": ["Fe"],
            "supercell_size": [1, 1, 1],
            "policy_name": "cold_start",
        },
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 500.0,
            "pseudopotentials": {"Fe": str(tmp_path / "Fe.UPF")},
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 5.0,
            "max_basis_size": 500,
        },
        "md": {
            "temperature": 300.0,
            "pressure": 0.0,
            "timestep": 0.001,
            "n_steps": 1000,
        },
        "workflow": {
            "max_iterations": 2,
            "state_file_path": str(tmp_path / "state.json"),
            "data_dir": str(tmp_path / "data"),
            "active_learning_dir": str(tmp_path / "active_learning"),
            "potentials_dir": str(tmp_path / "potentials"),
        },
        "validation": {
            "enabled": True,
            "phonon": {"supercell_size": [2, 2, 2], "displacement": 0.01},
            "elastic": {"strain_magnitude": 0.01},
        },
        "logging": {},
    }
    return PyAceConfig(**config_dict)


def test_scenario_07_01_validate_potential(uat_config: PyAceConfig, tmp_path: Path) -> None:
    """
    Scenario 07-01: Verify that the system can run a full validation suite on a potential.
    """
    # Setup Mocks
    # Important: ReportGenerator is instantiated inside Validator.__init__
    # We must patch the CLASS so that the instance used by Validator is a mock.
    with (
        patch("pyacemaker.core.validator.relax_structure") as mock_relax,
        patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon,
        patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic,
        patch("pyacemaker.core.validator.ReportGenerator") as MockReportGen,
    ):
        # Mock instances
        mock_phonon_calc = MockPhonon.return_value
        mock_elastic_calc = MockElastic.return_value

        # ReportGenerator mock
        mock_report_gen_instance = MockReportGen.return_value

        # Mock Returns
        relaxed_atoms = Atoms("Fe", cell=[2.8, 2.8, 2.8], pbc=True)
        mock_relax.return_value = relaxed_atoms

        mock_phonon_calc.calculate.return_value = PhononResult(
            has_imaginary_modes=False,
            status=ValidationStatus.PASS,
            band_structure_path=tmp_path / "bands.png",
        )

        mock_elastic_calc.calculate.return_value = ElasticResult(
            c_ij={"C11": 200.0, "C12": 100.0, "C44": 80.0},
            bulk_modulus=160.0,
            is_mechanically_stable=True,
            status=ValidationStatus.PASS,
        )

        # Execute
        validator = Validator(uat_config)

        # Verify that Validator is using our mocked ReportGenerator
        assert validator.report_generator == mock_report_gen_instance

        potential_path = tmp_path / "potential.yace"
        potential_path.touch()
        base_structure = Atoms("Fe", cell=[2.8, 2.8, 2.8], pbc=True)
        output_dir = tmp_path / "validation"

        report = validator.validate(potential_path, base_structure, output_dir)

        # Verify
        assert report.overall_status == ValidationStatus.PASS
        assert report.phonon
        assert report.phonon.status == ValidationStatus.PASS
        assert report.elastic
        assert report.elastic.status == ValidationStatus.PASS
        assert report.report_path == output_dir / DEFAULT_VALIDATION_REPORT_FILENAME

        # Verify calls
        mock_relax.assert_called_once()
        mock_phonon_calc.calculate.assert_called_once()
        mock_elastic_calc.calculate.assert_called_once()
        mock_report_gen_instance.generate.assert_called_once()


def test_scenario_07_02_unstable_detection(uat_config: PyAceConfig, tmp_path: Path) -> None:
    """
    Scenario 07-02: Verify that the system flags an unstable potential.
    """
    with (
        patch("pyacemaker.core.validator.relax_structure") as mock_relax,
        patch("pyacemaker.core.validator.PhononCalculator") as MockPhonon,
        patch("pyacemaker.core.validator.ElasticCalculator") as MockElastic,
        patch("pyacemaker.core.validator.ReportGenerator")
    ):
        mock_phonon_calc = MockPhonon.return_value
        mock_elastic_calc = MockElastic.return_value

        mock_relax.return_value = Atoms("Fe")

        # Simulate unstable phonons
        mock_phonon_calc.calculate.return_value = PhononResult(
            has_imaginary_modes=True,
            status=ValidationStatus.FAIL,
            band_structure_path=tmp_path / "bands.png",
        )

        # Simulate stable elastic (mixed results)
        mock_elastic_calc.calculate.return_value = ElasticResult(
            c_ij={"C11": 200.0},
            bulk_modulus=160.0,
            is_mechanically_stable=True,
            status=ValidationStatus.PASS,
        )

        validator = Validator(uat_config)
        report = validator.validate(Path("pot.yace"), Atoms("Fe"), tmp_path)

        assert report.overall_status == ValidationStatus.FAIL
        assert report.phonon
        assert report.phonon.status == ValidationStatus.FAIL
        assert report.elastic
        assert report.elastic.status == ValidationStatus.PASS
