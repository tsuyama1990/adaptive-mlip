from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.utils.io import (
    detect_elements,
    dump_yaml,
    load_config,
    write_lammps_streaming,
)


def test_load_config_valid() -> None:
    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text("key: value\n")
        config = load_config(config_path)
        assert config == {"key": "value"}


def test_load_config_invalid_yaml() -> None:
    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text("key: [unclosed list\n")
        import yaml
        with pytest.raises(yaml.YAMLError):
            load_config(config_path)


def test_dump_yaml() -> None:
    with TemporaryDirectory() as tmp_dir:
        dump_path = Path(tmp_dir) / "dump.yaml"
        data = {"key": "value"}
        dump_yaml(data, dump_path)
        assert dump_path.read_text() == "key: value\n"


def test_write_lammps_streaming() -> None:
    with TemporaryDirectory() as tmp_dir:
        dump_path = Path(tmp_dir) / "data.lmp"
        structure = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[5, 5, 5])
        with open(dump_path, "w") as f:
            write_lammps_streaming(f, structure, ["H"])

        content = dump_path.read_text()
        assert "2 atoms" in content
        assert "1 atom types" in content
        assert "Masses" in content
        assert "Atoms" in content


def test_detect_elements() -> None:
    from ase.io import write
    with TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir) / "data.xyz"
        structure = Atoms("H2O", positions=[[0,0,0], [1,0,0], [0,1,0]])
        write(str(data_path), structure)

        elements = detect_elements(data_path)
        assert elements == ["H", "O"]
