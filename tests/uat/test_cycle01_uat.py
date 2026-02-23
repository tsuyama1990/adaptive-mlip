import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


# Reusable fixture factory to reduce duplication
def create_config(tmp_path: Path, **overrides: Any) -> Path:
    defaults: dict[str, Any] = {
        "project_name": "FePt_Optimization",
        "structure": {"elements": ["Fe", "Pt"], "supercell_size": [2, 2, 2]},
        "dft": {"code": "qe", "functional": "PBE", "kpoints_density": 0.04, "encut": 500.0},
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 500},
        "md": {"temperature": 1000.0, "pressure": 0.0, "timestep": 0.001, "n_steps": 1000},
        "workflow": {"max_iterations": 10},
        "logging": {"log_file": "test.log"}
    }

    # Deep merge overrides would be better but for this simple case:
    for section, values in overrides.items():
        if section in defaults and isinstance(defaults[section], dict):
            # Explicitly cast to dict to satisfy mypy
            section_dict: dict[str, Any] = defaults[section]
            section_dict.update(values)
        else:
            defaults[section] = values

    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(defaults, f)
    return config_path

@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    return create_config(tmp_path)

@pytest.fixture
def invalid_config_file_temp(tmp_path: Path) -> Path:
    return create_config(tmp_path, project_name="Bad_Temp", md={"temperature": -100.0})

@pytest.fixture
def invalid_config_file_cutoff(tmp_path: Path) -> Path:
    return create_config(tmp_path, project_name="Bad_Cutoff", training={"cutoff_radius": -2.0})

def test_scenario_01_01_hello_config(valid_config_file: Path) -> None:
    env = {"PYTHONPATH": str(Path("src").resolve())}
    cwd = valid_config_file.parent
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", valid_config_file.name, "--dry-run"]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(cwd), check=False) # noqa: S603

    assert result.returncode == 0, f"Process failed. Stderr: {result.stderr}"
    assert "Configuration loaded successfully" in result.stdout or "Configuration loaded successfully" in result.stderr

def test_scenario_01_02_guardrails_check_temp(invalid_config_file_temp: Path) -> None:
    env = {"PYTHONPATH": str(Path("src").resolve())}
    cwd = invalid_config_file_temp.parent
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", invalid_config_file_temp.name]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(cwd), check=False)  # noqa: S603
    assert result.returncode != 0
    assert "validation error" in result.stderr
    assert "temperature" in result.stderr

def test_scenario_01_02_guardrails_check_cutoff(invalid_config_file_cutoff: Path) -> None:
    env = {"PYTHONPATH": str(Path("src").resolve())}
    cwd = invalid_config_file_cutoff.parent
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", invalid_config_file_cutoff.name]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(cwd), check=False)  # noqa: S603
    assert result.returncode != 0
    assert "validation error" in result.stderr
    assert "cutoff_radius" in result.stderr
