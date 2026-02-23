import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import create_test_config_dict


# Reusable fixture factory using shared utility
def create_config(tmp_path: Path, **overrides: Any) -> Path:
    config_dict = create_test_config_dict(**overrides)
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)
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

    # Use check=True for expected success
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(cwd), check=True)  # noqa: S603

    assert (
        "Configuration loaded successfully" in result.stdout
        or "Configuration loaded successfully" in result.stderr
    )


def test_scenario_01_02_guardrails_check_temp(invalid_config_file_temp: Path) -> None:
    env = {"PYTHONPATH": str(Path("src").resolve())}
    cwd = invalid_config_file_temp.parent
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", invalid_config_file_temp.name]

    # Expected failure, check=False is appropriate but we assert returncode != 0
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(cwd), check=False)  # noqa: S603
    assert result.returncode != 0
    assert "validation error" in result.stderr
    assert "temperature" in result.stderr


def test_scenario_01_02_guardrails_check_cutoff(invalid_config_file_cutoff: Path) -> None:
    env = {"PYTHONPATH": str(Path("src").resolve())}
    cwd = invalid_config_file_cutoff.parent
    cmd = [sys.executable, "-m", "pyacemaker.main", "--config", invalid_config_file_cutoff.name]

    # Expected failure
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(cwd), check=False)  # noqa: S603
    assert result.returncode != 0
    assert "validation error" in result.stderr
    assert "cutoff_radius" in result.stderr
