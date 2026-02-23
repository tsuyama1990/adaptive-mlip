from pathlib import Path

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

    loaded = load_yaml(p)
    assert loaded == data


def test_load_yaml_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml("non_existent_file.yaml")


def test_load_yaml_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "invalid.yaml"
    with p.open("w") as f:
        f.write("{unclosed_brace: value")

    with pytest.raises(ValueError, match="Error parsing YAML"):
        load_yaml(p)


def test_load_yaml_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.touch()
    assert load_yaml(p) == {}


def test_load_yaml_not_dict(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    with p.open("w") as f:
        f.write("- item1\n- item2")

    with pytest.raises(ValueError, match="must contain a dictionary"):
        load_yaml(p)


def test_load_yaml_directory(tmp_path: Path) -> None:
    p = tmp_path / "subdir"
    p.mkdir()
    with pytest.raises(ValueError, match="Path is not a file"):
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

    with pytest.raises(ValidationError):
        load_config(p)
