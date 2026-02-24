from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyacemaker.main import main


@pytest.fixture
def scenario_config(tmp_path: Path) -> Path:
    config_data = {
        "project_name": "test_scenario",
        "structure": {
            "elements": ["Fe", "Pt", "Mg", "O"],
            "supercell_size": [1, 1, 1],
        },
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 400,
            "pseudopotentials": {"Fe": "Fe.pbe-n-kjpaw_psl.1.0.0.UPF"},
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 4.0,
            "max_basis_size": 500,
        },
        "md": {
            "temperature": 300.0,
            "pressure": 1.0,
            "n_steps": 1000,
            "timestep": 0.001,
            "thermo_freq": 100,
            "dump_freq": 100,
        },
        "workflow": {"max_iterations": 1},
        "scenario": {
            "name": "fept_mgo",
            "enabled": True,
            "parameters": {"steps": 10, "potential_path": str(tmp_path / "pot.yace")},
        },
        "eon": {
            "enabled": True,
            "potential_path": str(tmp_path / "pot.yace"),
            "akmc_steps": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    # Create dummy potential file
    (tmp_path / "pot.yace").touch()
    (tmp_path / "structure.xyz").touch()
    (tmp_path / "pseudos").mkdir()
    (tmp_path / "Fe.pbe-n-kjpaw_psl.1.0.0.UPF").touch()  # Create at root because config says: {"Fe": "..."}
    # Wait, config says pseudopotentials: {"Fe": "Fe.pbe-n-kjpaw_psl.1.0.0.UPF"}
    # But dft schema might prepend pseudopotentials_dir if present?
    # No, I removed pseudopotentials_dir from config above because it was forbidden extra.
    # So path is relative to CWD.

    return config_path


def test_main_runs_scenario(scenario_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure CWD matches config location to satisfy load_yaml security check
    monkeypatch.chdir(scenario_config.parent)

    # We patch the specific scenario class to verify it's called
    with patch("pyacemaker.main.get_scenario_runner") as mock_runner_factory:
        mock_runner = MagicMock()
        mock_runner_factory.return_value = mock_runner

        # Simulate CLI arguments
        # Note: We must use the filename only since we chdir'd
        with patch("sys.argv", ["main.py", "--config", scenario_config.name, "--scenario", "fept_mgo"]):
            main()

        mock_runner_factory.assert_called_once()
        mock_runner.run.assert_called_once()
