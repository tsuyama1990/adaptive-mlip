
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
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
    assert lines[0] == "LAMMPS data file written by pyacemaker (streaming)"
    assert lines[1] == ""
    assert lines[2] == "2 atoms"
    assert lines[3] == "2 atom types"
    assert lines[4] == ""
    assert lines[5] == "0.000000 4.000000 xlo xhi"
    assert lines[6] == "0.000000 4.000000 ylo yhi"
    assert lines[7] == "0.000000 4.000000 zlo zhi"
    assert lines[8] == ""
    assert lines[9] == "Masses"
    assert lines[10] == ""
    # H is type 1
    assert "1 1.0080 # H" in content
    # Li is type 2
    assert "2 6.9400 # Li" in content

    assert "Atoms" in content

    # Atom lines: id type x y z
    # Li (index 0 in atoms) -> type 2. pos 0 0 0
    # H (index 1 in atoms) -> type 1. pos 0.5 0.5 0.5

    # Since dict iteration order is insertion order in modern python, and we iterate atoms...
    # Atom 1 is Li -> 1 2 ...
    assert "1 2 0.000000 0.000000 0.000000" in content
    assert "2 1 0.500000 0.500000 0.500000" in content

def test_write_lammps_streaming_large_structure_mock() -> None:
    """
    Verifies that write_lammps_streaming can handle a large number of atoms
    without crashing or errors, using a mock structure to avoid memory overhead.
    """
    # Removed unused buffer = StringIO()

    with patch("ase.Atoms.__len__", return_value=1000), \
         patch.object(Atoms, "get_positions", return_value=np.zeros((1000, 3))), \
         patch.object(Atoms, "get_chemical_symbols", return_value=["H"]*1000), \
         patch.object(Atoms, "get_cell", return_value=np.eye(3)):

         structure = Atoms("H", positions=[[0,0,0]], cell=[10,10,10])
         # Use a mock file object to count writes
         mock_file = MagicMock()
         write_lammps_streaming(mock_file, structure, ["H"])

         # Header writes + 1000 atom lines + mass lines + box lines
         # 1000 atoms -> 1000 calls for atoms
         assert mock_file.write.call_count > 1000
