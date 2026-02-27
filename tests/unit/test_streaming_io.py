from collections.abc import Iterable, Iterator
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.utils.io import write_lammps_streaming


def test_write_lammps_streaming_basic() -> None:
    """Test basic streaming write functionality."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10])
    buffer = StringIO()

    write_lammps_streaming(buffer, atoms, species=["H"])

    content = buffer.getvalue()
    assert "LAMMPS data file via pyacemaker streaming" in content
    assert "2 atoms" in content
    assert "1 atom types" in content
    assert "0.000000 10.000000 xlo xhi" in content
    assert "1 1 0.000000 0.000000 0.000000" in content


def test_write_lammps_streaming_io_failure() -> None:
    """Test handling of I/O failure during streaming."""
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10])

    # Mock file object that raises IOError on write
    mock_file = MagicMock()
    # Mock both write and writelines as optimized version uses writelines
    mock_file.write.side_effect = IOError("Disk full")
    mock_file.writelines.side_effect = IOError("Disk full")

    with pytest.raises(IOError, match="Disk full"):
        write_lammps_streaming(mock_file, atoms, species=["H"])


def test_write_lammps_streaming_invalid_species() -> None:
    """Test handling of atoms with species not in the list."""
    atoms = Atoms("He", positions=[[0, 0, 0]], cell=[10, 10, 10])
    buffer = StringIO()

    with pytest.raises(KeyError, match="Symbol not in provided species list"):
        write_lammps_streaming(buffer, atoms, species=["H"])


def test_write_lammps_streaming_non_orthogonal_check() -> None:
    """Test rejection of non-orthogonal cells."""
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[[10, 1, 0], [0, 10, 0], [0, 0, 10]])
    buffer = StringIO()

    with pytest.raises(ValueError, match="Streaming write currently only supports orthogonal cells"):
        write_lammps_streaming(buffer, atoms, species=["H"])


def test_streaming_memory_usage() -> None:
    """
    Test that large structures are handled in chunks (implied by implementation).
    """
    # Create "large" system
    n_atoms = 2500 # > 1000 chunk size
    atoms = Atoms("H" * n_atoms, positions=np.zeros((n_atoms, 3)), cell=[100, 100, 100])

    mock_file = MagicMock()

    write_lammps_streaming(mock_file, atoms, species=["H"])

    # Check writelines calls
    writelines_calls = mock_file.writelines.call_args_list
    assert len(writelines_calls) >= 3 # 2500 atoms / 1000 chunk = 3 chunks (1000, 1000, 500)

    # Verify first chunk size
    first_chunk = writelines_calls[0][0][0]
    assert len(first_chunk) == 1000
