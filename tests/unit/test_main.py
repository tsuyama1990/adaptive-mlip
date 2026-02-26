from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyacemaker.main import main


@pytest.fixture
def mock_config(tmp_path: Path):
    config = {
        "project_name": "TestProject",
        "structure": {"elements": ["Al"], "supercell_size": [1, 1, 1], "r_cut": 2.0},
        "dft": {
            "code": "qe", "functional": "PBE", "kpoints_density": 0.04, "encut": 500,
            "pseudopotentials": {"Al": "Al.UPF"}
        },
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 100},
        "md": {"temperature": 300, "pressure": 0, "timestep": 0.001, "n_steps": 1000},
        "workflow": {"active_learning_strategy": "max_uncertainty", "max_iterations": 5}
    }
    p = tmp_path / "config.yaml"
    with p.open("w") as f:
        yaml.dump(config, f)
    return p

def test_main_init_success(mock_config: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test 'init' command successfully initializes workspace."""
    monkeypatch.setattr("sys.argv", ["pyacemaker", "init", "--config", str(mock_config)])

    with (
        patch("pyacemaker.orchestrator.Orchestrator.initialize_workspace") as mock_init,
        patch("pyacemaker.main.PyAceConfig") as mock_conf_cls,
    ):
         config_mock = MagicMock()
         config_mock.project_name = "TestProject"
         # Use configure_mock for nested attributes to ensure they persist
         config_mock.workflow.data_dir = tmp_path / "data"
         config_mock.workflow.potentials_dir = tmp_path / "potentials"
         config_mock.workflow.active_learning_dir = tmp_path / "al_dir"
         config_mock.logging.log_dir = tmp_path / "logs"
         config_mock.workflow.state_file_path = tmp_path / "state.json"
         config_mock.structure.num_structures = 10

         mock_conf_cls.return_value = config_mock
         main()
         mock_init.assert_called_once()

def test_main_init_file_not_found():
    """Test 'init' command with missing config file."""
    with patch("sys.argv", ["pyacemaker", "init", "--config", "non_existent.yaml"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

def test_main_run_step1(mock_config: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test 'run' command calls run_step1."""
    monkeypatch.setattr("sys.argv", ["pyacemaker", "run", "--step", "1", "--config", str(mock_config)])

    with (
        patch("pyacemaker.orchestrator.Orchestrator.run_step1") as mock_run1,
        patch("pyacemaker.main.PyAceConfig") as mock_conf_cls,
    ):
         config_mock = MagicMock()
         config_mock.project_name = "TestProject"
         config_mock.workflow.data_dir = tmp_path / "data"
         config_mock.workflow.potentials_dir = tmp_path / "potentials"
         config_mock.workflow.active_learning_dir = tmp_path / "al_dir"
         config_mock.logging.log_dir = tmp_path / "logs"
         config_mock.workflow.state_file_path = tmp_path / "state.json"

         mock_conf_cls.return_value = config_mock
         main()
         mock_run1.assert_called_once()

def test_main_run_no_config(monkeypatch: pytest.MonkeyPatch):
    """Test 'run' without config argument (fails if default not found)."""
    # Assuming local config.yaml doesn't exist in CWD (which is set by pytest usually)
    monkeypatch.setattr("sys.argv", ["pyacemaker", "run", "--step", "1"])

    # We expect failure because "config.yaml" likely doesn't exist in test execution CWD unless mocked
    # We'll check if it tries to open "config.yaml"
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(SystemExit) as exc:
             main()
        assert exc.value.code == 1
