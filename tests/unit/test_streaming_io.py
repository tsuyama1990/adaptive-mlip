from io import StringIO

import pytest
from ase import Atoms

from pyacemaker.utils.io import write_lammps_streaming


def test_write_lammps_streaming_simple() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
    elements = ["H"]

    buffer = StringIO()
    write_lammps_streaming(buffer, atoms, elements)

    content = buffer.getvalue()
    assert "2 atoms" in content
    assert "1 atom types" in content
    assert "0.000000 10.000000 xlo xhi" in content
    assert "Masses" in content
    assert "Atoms" in content
    # Check atom lines
    # id type x y z
    assert "1 1 0.000000 0.000000 0.000000" in content
    assert "2 1 1.000000 0.000000 0.000000" in content


def test_write_lammps_streaming_multiple_types() -> None:
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[10, 10, 10], pbc=True)
    elements = ["C", "O"]  # Sorted order: C, O

    buffer = StringIO()
    write_lammps_streaming(buffer, atoms, elements)

    content = buffer.getvalue()
    assert "2 atom types" in content
    # C is type 1, O is type 2
    assert "1 1 0.000000 0.000000 0.000000" in content
    assert "2 2 1.200000 0.000000 0.000000" in content


def test_write_lammps_streaming_non_orthogonal() -> None:
    atoms = Atoms("H", cell=[[10, 1, 0], [0, 10, 0], [0, 0, 10]], pbc=True)
    elements = ["H"]
    buffer = StringIO()

    with pytest.raises(ValueError, match="Streaming writer only supports orthogonal"):
        write_lammps_streaming(buffer, atoms, elements)
