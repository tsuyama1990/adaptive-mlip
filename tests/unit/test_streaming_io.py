from io import StringIO

import pytest
from ase import Atoms

from pyacemaker.utils.io import write_lammps_streaming


def test_write_lammps_streaming_format() -> None:
    """Verifies that write_lammps_streaming produces correct LAMMPS data format."""
    buffer = StringIO()

    # Create simple structure: 2 atoms, cubic box
    structure = Atoms("LiH", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[4.0, 4.0, 4.0], pbc=True)
    elements = ["H", "Li"] # Sorted order: H (Z=1), Li (Z=3)

    write_lammps_streaming(buffer, structure, elements)

    content = buffer.getvalue()
    lines = content.splitlines()

    # Verify exact format
    assert lines[0] == "LAMMPS data file via pyacemaker streaming"
    assert lines[1] == ""
    assert lines[2] == "2 atoms"
    assert lines[3] == "2 atom types"
    assert lines[4] == ""
    # Note: float formatting might vary slightly, checking start/end
    assert "xlo xhi" in lines[5]
    assert "ylo yhi" in lines[6]
    assert "zlo zhi" in lines[7]
    assert lines[8] == ""
    assert lines[9] == "Masses"
    assert lines[10] == ""

    # Check masses content roughly
    # H (type 1) ~ 1.008
    # Li (type 2) ~ 6.94
    assert "1 1.00" in content
    assert "# H" in content
    assert "2 6.94" in content
    assert "# Li" in content

    assert "Atoms" in content

    # Atom lines: id type x y z
    # Li (index 0 in atoms) -> type 2. pos 0 0 0
    # H (index 1 in atoms) -> type 1. pos 0.5 0.5 0.5

    # Since dict iteration order is insertion order in modern python, and we iterate atoms...
    # Atom 1 is Li -> 1 2 ...
    assert "1 2 0.000000 0.000000 0.000000" in content
    assert "2 1 0.500000 0.500000 0.500000" in content


def test_write_lammps_streaming_invalid_elements() -> None:
    """Test validation of missing elements."""
    buffer = StringIO()
    structure = Atoms("LiH", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[4.0, 4.0, 4.0], pbc=True)
    # Missing Li in elements list
    elements = ["H"]

    with pytest.raises(KeyError, match="not in provided species list"):
        write_lammps_streaming(buffer, structure, elements)


def test_write_lammps_streaming_non_orthogonal() -> None:
    """Test validation of non-orthogonal cells (not supported by simple streaming yet)."""
    buffer = StringIO()
    # Non-orthogonal cell
    cell = [[10, 0, 0], [5, 8.66, 0], [0, 0, 10]]
    structure = Atoms("H", positions=[[0, 0, 0]], cell=cell, pbc=True)
    elements = ["H"]

    with pytest.raises(ValueError, match="Streaming write currently only supports orthogonal cells"):
        write_lammps_streaming(buffer, structure, elements)
