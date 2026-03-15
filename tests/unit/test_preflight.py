from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.build import bulk

from pyacemaker.core.preflight import (
    DependencyValidator,
    LammpsSyntaxValidator,
    PreflightManager,
    StructuralValidator,
)
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.preflight import DiagnosticReport, Severity


@pytest.fixture
def clean_report() -> DiagnosticReport:
    return DiagnosticReport()


@pytest.fixture
def mock_config() -> PyAceConfig:
    return PyAceConfig(
        project_name="TestProject",
        structure={
            "elements": ["Al"],
            "supercell_size": [1, 1, 1],
            "policy_name": "cold_start"
        },
        dft={
            "code": "qe",
            "functional": "pbe",
            "kpoints_density": 2.0,
            "encut": 40.0,
            "pseudopotentials": {"Al": "fake.UPF"},
            "mixing_beta": 0.7,
            "smearing_type": "gaussian",
            "smearing_width": 0.1,
            "diagonalization": "david"
        },
        training={
            "potential_type": "mace",
            "cutoff_radius": 5.0,
            "max_basis_size": 200,
            "delta_learning": False,
            "active_set_optimization": False,
            "foundation_model_path": "fake.model",
            "pacemaker": {}
        },
        md={
            "temperature": 300.0,
            "pressure": 0.0,
            "timestep": 0.001,
            "n_steps": 1000,
            "uncertainty_threshold": 0.1,
            "check_interval": 10
        },
        workflow={
            "max_iterations": 2,
            "state_file_path": "state.json",
            "data_dir": "data",
            "active_learning_dir": "al",
            "potentials_dir": "pot"
        },
        logging={}
    )


def test_structural_validator_clean(mock_config: PyAceConfig, clean_report: DiagnosticReport) -> None:
    validator = StructuralValidator()

    with patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_create:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = iter([bulk("Al", "fcc", a=4.0)])
        mock_create.return_value = (mock_gen, None, None, None, None, None)
        validator.validate(mock_config, clean_report)

    assert len(clean_report.errors) == 0


def test_structural_validator_collision(mock_config: PyAceConfig, clean_report: DiagnosticReport) -> None:
    validator = StructuralValidator()

    # Create structure with collision
    atoms = Atoms("Al2", positions=[[0,0,0], [0.1, 0, 0]], cell=[5,5,5], pbc=True)

    with patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_create:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = iter([atoms])
        mock_create.return_value = (mock_gen, None, None, None, None, None)
        validator.validate(mock_config, clean_report)

    assert len(clean_report.errors) == 1
    assert "Atomic collision detected" in clean_report.errors[0].description
    assert clean_report.errors[0].severity == Severity.ERROR


def test_dependency_validator_missing_files(mock_config: PyAceConfig, clean_report: DiagnosticReport) -> None:
    validator = DependencyValidator()

    # pw.x missing, UPF missing, model missing
    with patch("shutil.which", return_value=None), patch.object(Path, "exists", return_value=False):
        validator.validate(mock_config, clean_report)

    # Expected 3 errors: pw.x, mace_run_train, fake.UPF, fake.model
    # Because mace_run_train is added when potential_type is mace
    assert len(clean_report.errors) == 4
    err_texts = [e.description for e in clean_report.errors]
    assert any("pw.x" in t for t in err_texts)
    assert any("fake.UPF" in t for t in err_texts)
    assert any("fake.model" in t for t in err_texts)


def test_lammps_syntax_validator_clean(mock_config: PyAceConfig, clean_report: DiagnosticReport) -> None:
    validator = LammpsSyntaxValidator()

    atoms = bulk("Al", "fcc", a=4.0)

    with patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_create:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = iter([atoms])
        mock_create.return_value = (mock_gen, None, None, None, None, None)

        # Mock subprocess to return success
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            validator.validate(mock_config, clean_report)

    assert len(clean_report.errors) == 0


def test_lammps_syntax_validator_error(mock_config: PyAceConfig, clean_report: DiagnosticReport) -> None:
    validator = LammpsSyntaxValidator()

    atoms = bulk("Al", "fcc", a=4.0)

    with patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_create:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = iter([atoms])
        mock_create.return_value = (mock_gen, None, None, None, None, None)

        # Mock subprocess to return failure with error message
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "ERROR: Unknown fix style nve_typo (src/modify.cpp:123)\n"

        with patch("subprocess.run", return_value=mock_proc):
            validator.validate(mock_config, clean_report)

    assert len(clean_report.errors) == 1
    assert "Unknown fix style nve_typo" in clean_report.errors[0].description


def test_preflight_manager_success(mock_config: PyAceConfig) -> None:
    manager = PreflightManager()

    with patch.object(StructuralValidator, "validate") as m_str, \
         patch.object(DependencyValidator, "validate") as m_dep, \
         patch.object(LammpsSyntaxValidator, "validate") as m_lmp:

        report = manager.run(mock_config)

        assert m_str.called
        assert m_dep.called
        assert m_lmp.called

        assert len(report.errors) == 0
        assert len(report.info) == 1
        assert "successfully" in report.info[0].description
