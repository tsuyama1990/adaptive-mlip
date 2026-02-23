import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# Fixtures to create temporary config files
@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    config_data = {
        "project_name": "FePt_Optimization",
        "structure": {"elements": ["Fe", "Pt"], "supercell_size": [2, 2, 2]},
        "dft": {"code": "qe", "functional": "PBE", "kpoints_density": 0.04, "encut": 500.0},
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 500},
        "md": {"temperature": 1000.0, "pressure": 0.0, "timestep": 0.001, "n_steps": 1000},
        "workflow": {"max_iterations": 10},
    }
    config_path = tmp_path / "valid_config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path

@pytest.fixture
def invalid_config_file_temp(tmp_path: Path) -> Path:
    config_data = {
        "project_name": "Bad_Config",
        "structure": {"elements": ["Fe"], "supercell_size": [1, 1, 1]},
        "dft": {"code": "qe", "functional": "PBE", "kpoints_density": 0.04, "encut": 500.0},
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 500},
        "md": {"temperature": -100.0, "pressure": 0.0, "timestep": 0.001, "n_steps": 1000},  # Invalid
        "workflow": {"max_iterations": 10},
    }
    config_path = tmp_path / "bad_config_temp.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def invalid_config_file_cutoff(tmp_path: Path) -> Path:
    config_data = {
        "project_name": "Bad_Config",
        "structure": {"elements": ["Fe"], "supercell_size": [1, 1, 1]},
        "dft": {"code": "qe", "functional": "PBE", "kpoints_density": 0.04, "encut": 500.0},
        "training": {"potential_type": "ace", "cutoff_radius": -2.0, "max_basis_size": 500},  # Invalid
        "md": {"temperature": 300.0, "pressure": 0.0, "timestep": 0.001, "n_steps": 1000},
        "workflow": {"max_iterations": 10},
    }
    config_path = tmp_path / "bad_config_cutoff.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    return config_path


def test_scenario_01_01_hello_config(valid_config_file: Path) -> None:
    """
    Scenario 01-01: 'Hello Config'
    Objective: Verify that the system can read a standard configuration file and start up without errors.
    """
    # Run the main module with dry-run
    # Note: Using python -m pyacemaker.main might fail if the package is not installed in editable mode or PYTHONPATH is not set.
    # So we set PYTHONPATH to src/
    env = {"PYTHONPATH": "src"}

    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", str(valid_config_file), "--dry-run"]

    try:
        # S603 is ignored as we are running our own trusted code
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)  # noqa: S603
        assert result.returncode == 0
        assert (
            "Configuration loaded successfully" in result.stdout
            or "Configuration loaded successfully" in result.stderr
        )
        assert (
            "Project: FePt_Optimization initialized" in result.stdout
            or "Project: FePt_Optimization initialized" in result.stderr
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Process failed with exit code {e.returncode}. Output: {e.stdout}\nError: {e.stderr}"
        )
    except FileNotFoundError:
        pytest.fail("python executable not found")


def test_scenario_01_02_guardrails_check_temp(invalid_config_file_temp: Path) -> None:
    """
    Scenario 01-02: 'Guardrails Check' - Negative Temperature
    """
    env = {"PYTHONPATH": "src"}
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", str(invalid_config_file_temp)]

    # We expect non-zero exit code
    # S603 ignored, check=False explicit
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)  # noqa: S603
    assert result.returncode != 0
    assert "validation error" in result.stderr
    assert "temperature" in result.stderr


def test_scenario_01_02_guardrails_check_cutoff(invalid_config_file_cutoff: Path) -> None:
    """
    Scenario 01-02: 'Guardrails Check' - Negative Cutoff
    """
    env = {"PYTHONPATH": "src"}
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", str(invalid_config_file_cutoff)]

    # We expect non-zero exit code
    # S603 ignored, check=False explicit
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)  # noqa: S603
    assert result.returncode != 0
    assert "validation error" in result.stderr
    assert "cutoff_radius" in result.stderr
