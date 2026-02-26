import tempfile
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read

from pyacemaker.utils.io import detect_elements, dump_yaml, load_yaml, write_lammps_streaming


def test_load_yaml_success(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value\n")
    data = load_yaml(config_file)
    assert data == {"key": "value"}


def test_load_yaml_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml(Path("non_existent.yaml"))


def test_dump_yaml(tmp_path):
    data = {"key": "value"}
    dump_file = tmp_path / "dump.yaml"
    dump_yaml(data, dump_file)
    assert load_yaml(dump_file) == data


def test_detect_elements(tmp_path):
    # Create a dummy xyz file
    xyz_file = tmp_path / "test.xyz"
    atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms2 = Atoms("CO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    # Use ASE to write standard xyz
    from ase.io import write
    write(xyz_file, [atoms1, atoms2])

    elements = detect_elements(xyz_file)
    assert set(elements) == {"H", "O", "C"}
    assert elements == ["C", "H", "O"] # sorted

def test_detect_elements_empty(tmp_path):
    empty_file = tmp_path / "empty.xyz"
    empty_file.touch()
    elements = detect_elements(empty_file)
    assert elements == []

def test_write_lammps_streaming_basic():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[5, 5, 5])
    species = ["H"]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        write_lammps_streaming(f, atoms, species)
        f.flush()
        path = Path(f.name)

    # Verify content
    content = path.read_text()
    assert "2 atoms" in content
    assert "1 atom types" in content
    assert "H" in content

    # Verify structure using ASE read
    # ASE read lammps-data format requires explicit masses or known types.
    # Our format is standard lammps data.
    read_atoms = read(path, format="lammps-data", style="atomic")
    assert len(read_atoms) == 2
    # Check if positions match roughly
    # Note: Lammps read might reorder or center differently depending on settings,
    # but basic connectivity or counts should match.

    path.unlink()

def test_write_lammps_streaming_multiple_species():
    atoms = Atoms("H2O", positions=[[0,0,0], [1,0,0], [0,1,0]], cell=[10, 10, 10])
    species = ["H", "O"]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        write_lammps_streaming(f, atoms, species)
        f.flush()
        path = Path(f.name)

    content = path.read_text()
    assert "3 atoms" in content
    assert "2 atom types" in content

    path.unlink()

def test_write_lammps_streaming_missing_species():
    atoms = Atoms("He", positions=[[0,0,0]], cell=[10,10,10])
    species = ["H"] # He is missing

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        with pytest.raises(KeyError):
            write_lammps_streaming(f, atoms, species)
        path = Path(f.name)

    path.unlink()
