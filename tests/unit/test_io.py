from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.utils.io import load_config, load_yaml


def test_load_yaml_valid(tmp_path: Path) -> None:
    data = {"key": "value"}
    p = tmp_path / "test.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)

    # Mock cwd to be tmp_path so check passes
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        loaded = load_yaml(p)
    assert loaded == data


def test_load_yaml_file_not_found(tmp_path: Path) -> None:
    with patch("pathlib.Path.cwd", return_value=tmp_path), pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "non_existent_file.yaml")


def test_load_yaml_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "invalid.yaml"
    with p.open("w") as f:
        f.write("{unclosed_brace: value")

    with (
        patch("pathlib.Path.cwd", return_value=tmp_path),
        pytest.raises(ValueError, match="Error parsing YAML"),
    ):
        load_yaml(p)


def test_load_yaml_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.touch()
    # Combine patch and pytest.raises into single with statement
    # However, Python < 3.9 doesn't support parenthesized context managers cleanly
    # if we strictly follow older styles, but we target >=3.11.
    # The SIM117 suggests combining nested.

    with (
        patch("pathlib.Path.cwd", return_value=tmp_path),
        pytest.raises(ValueError, match="YAML file is empty"),
    ):
        load_yaml(p)


def test_load_yaml_not_dict(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    with p.open("w") as f:
        f.write("- item1\n- item2")

    # Updated to TypeError for TRY004
    with (
        patch("pathlib.Path.cwd", return_value=tmp_path),
        pytest.raises(TypeError, match="must contain a dictionary"),
    ):
        load_yaml(p)


def test_load_yaml_directory(tmp_path: Path) -> None:
    p = tmp_path / "subdir"
    p.mkdir()
    with (
        patch("pathlib.Path.cwd", return_value=tmp_path),
        pytest.raises(ValueError, match="Path is not a file"),
    ):
        load_yaml(p)


def test_load_config_valid(tmp_path: Path) -> None:
    config_data = {
        "project_name": "Test",
        "structure": {"elements": ["Fe"], "supercell_size": [1, 1, 1]},
        "dft": {"code": "qe", "functional": "PBE", "kpoints_density": 0.04, "encut": 500.0},
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 500},
        "md": {"temperature": 300.0, "pressure": 0.0, "timestep": 0.001, "n_steps": 1000},
        "workflow": {"max_iterations": 10},
    }
    p = tmp_path / "config.yaml"
    with p.open("w") as f:
        yaml.dump(config_data, f)

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        config = load_config(p)
    assert isinstance(config, PyAceConfig)
    assert config.project_name == "Test"


def test_load_config_invalid(tmp_path: Path) -> None:
    config_data = {
        "project_name": "Test",
        # Missing structure
    }
    p = tmp_path / "invalid_config.yaml"
    with p.open("w") as f:
        yaml.dump(config_data, f)

    with patch("pathlib.Path.cwd", return_value=tmp_path), pytest.raises(ValidationError):
        load_config(p)


def test_path_traversal_check(tmp_path: Path) -> None:
    # Create a file outside the "current working directory"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    p = outside_dir / "config.yaml"
    p.touch()

    # Set CWD to a different subdir
    cwd = tmp_path / "inside"
    cwd.mkdir()

    with (
        patch("pathlib.Path.cwd", return_value=cwd),
        pytest.raises(ValueError, match="Path traversal detected"),
    ):
        load_yaml(p)
