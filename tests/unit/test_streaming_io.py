
from unittest.mock import MagicMock, patch
from io import StringIO

import pytest
import numpy as np
from ase import Atoms

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.utils.io import write_lammps_streaming


@pytest.fixture
def mock_config(tmp_path):
    # Helper to create a minimal config
    # Need to create dummy pseudopotential for strict validation
    pp_path = tmp_path / "Fe.UPF"
    pp_path.write_text("<UPF version='2.0.1'>")

    from tests.conftest import create_test_config_dict
    config_dict = create_test_config_dict(
        dft={"pseudopotentials": {"Fe": str(pp_path)}}
    )
    return PyAceConfig(**config_dict)

@pytest.fixture
def orchestrator(mock_config, tmp_path):
    # Update config paths to tmp_path
    mock_config.workflow.state_file_path = str(tmp_path / "state.json")
    mock_config.workflow.data_dir = str(tmp_path / "data")
    mock_config.workflow.active_learning_dir = str(tmp_path / "active_learning")
    mock_config.workflow.potentials_dir = str(tmp_path / "potentials")
    return Orchestrator(mock_config)

def test_stream_write_chunking(orchestrator, tmp_path):
    """Verifies that _stream_write processes items in chunks."""
    output_file = tmp_path / "stream_output.xyz"

    # Create a generator of 25 items
    atoms_list = [Atoms("H") for _ in range(25)]
    atoms_gen = (a for a in atoms_list)

    # Batch size 10 -> Should call write 3 times (10, 10, 5)
    batch_size = 10

    # Mock ase.io.write
    with patch("pyacemaker.orchestrator.write") as mock_write:
        count = orchestrator._stream_write(atoms_gen, output_file, batch_size=batch_size)

        assert count == 25
        assert mock_write.call_count == 3

        # Check call args
        # Should call write(f, batch, format="extxyz") 3 times.
        # Since we pass file handle, append kwarg is NOT used.
        # Mode is controlled by filepath.open() which we didn't mock here, so it opened real file.

        args1, kwargs1 = mock_write.call_args_list[0]
        # args1[0] is file handle (not easily checked), args1[1] is batch
        assert len(args1[1]) == 10
        assert kwargs1.get("format") == "extxyz"
        assert "append" not in kwargs1

        args2, kwargs2 = mock_write.call_args_list[1]
        assert len(args2[1]) == 10

        args3, kwargs3 = mock_write.call_args_list[2]
        assert len(args3[1]) == 5

def test_stream_write_append_mode(orchestrator, tmp_path):
    """Verifies behavior when append=True is passed."""
    output_file = tmp_path / "append_output.xyz"
    output_file.touch() # Exists

    atoms_list = [Atoms("H") for _ in range(5)]
    atoms_gen = (a for a in atoms_list)

    # Mock open to verify mode
    with patch("pathlib.Path.open") as mock_open:
        # We need mock_open to return context manager
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch("pyacemaker.orchestrator.write") as mock_write:
            orchestrator._stream_write(atoms_gen, output_file, batch_size=2, append=True)

            # Verify mode='a'
            mock_open.assert_called_with("a")

            # Verify write calls
            assert mock_write.call_count == 3

def test_write_lammps_streaming_format() -> None:
    """Verifies that write_lammps_streaming produces correct LAMMPS data format."""
    buffer = StringIO()

    # Create simple structure: 2 atoms, cubic box
    structure = Atoms("LiH", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[4.0, 4.0, 4.0], pbc=True)
    elements = ["H", "Li"] # Sorted order: H (Z=1), Li (Z=3)

    write_lammps_streaming(buffer, structure, elements)

    content = buffer.getvalue()

    # Check Header
    assert "LAMMPS data file" in content
    assert "2 atoms" in content
    assert "2 atom types" in content
    assert "0.000000 4.000000 xlo xhi" in content

    # Check Masses
    assert "Masses" in content
    # H (1) -> 1.008, Li (3) -> 6.94
    # element list order: H=1, Li=2 in LAMMPS types
    # But usually type ID matches order in elements list
    # H is first in 'elements' -> type 1
    # Li is second -> type 2
    assert "1 1.0080" in content # H
    assert "2 6.94" in content # Li

    # Check Atoms
    assert "Atoms" in content
    # Atom 1: Li. Li is type 2.
    # Atom 2: H. H is type 1.
    # Format: id type x y z
    assert "1 2 0.000000 0.000000 0.000000" in content
    assert "2 1 0.500000 0.500000 0.500000" in content

def test_write_lammps_streaming_non_orthogonal() -> None:
    """Verifies that non-orthogonal cells raise an error."""
    buffer = StringIO()
    structure = Atoms("H", cell=[[1, 0, 0], [0.5, 1, 0], [0, 0, 1]]) # Triclinic
    elements = ["H"]

    with pytest.raises(ValueError, match="Streaming writer only supports orthogonal cells"):
        write_lammps_streaming(buffer, structure, elements)

def test_write_lammps_streaming_large_structure_mock() -> None:
    """
    Verifies that write_lammps_streaming can handle a large number of atoms
    without crashing or errors, using a mock structure to avoid memory overhead.
    """
    buffer = StringIO()

    # Create a mock structure that reports 1 million atoms but doesn't store them all
    # This tests the loop logic but depends on ASE Atoms implementation details.
    # Instead, we construct a real but lightweight Atoms object and patch get_positions/symbols

    n_atoms = 1_000_000

    # Mocking is tricky because Atoms is complex.
    # We will create a small structure but 'pretend' it has many atoms by patching len()
    # and get_positions/get_chemical_symbols to return iterators/generators if write_lammps_streaming supported them.
    # But write_lammps_streaming currently accesses arrays directly:
    # positions = structure.get_positions()
    # symbols = structure.get_chemical_symbols()

    # If we want to support streaming INPUT structure, write_lammps_streaming needs to accept iterables.
    # Currently it takes `structure: Atoms`. Atoms holds data in memory.
    # So `write_lammps_streaming` prevents *additional* copies, but input must be in memory if it's an Atoms object.
    # The requirement "NEVER load entire datasets into memory" implies we should stream FROM disk TO disk.
    # `io_manager.py` does: `write_lammps_streaming(f, structure, elements)`. `structure` is passed in.
    # `prepare_workspace(self, structure: Atoms)`.
    # So the *input* to `prepare_workspace` is already an in-memory `Atoms` object!

    # If the caller loads the whole file into `structure`, we already failed "NEVER load entire datasets".
    # However, `prepare_workspace` is called by `LammpsEngine.run`.
    # `Orchestrator` calls `engine.run(candidates[i])`. Candidates are loaded one by one.
    # So `structure` is a single configuration.
    # A single configuration of 1M atoms is large but might fit in memory (1M * 3 * 8 bytes ~ 24MB).
    # The "Memory Safety Violation" might refer to loading a *trajectory* of 1M frames.
    # Or creating copies of the 1M atoms structure.

    # `write_lammps_streaming` avoids creating the formatted string for the whole file in memory.
    # Standard `ase.io.write` might build a huge string buffer.

    # To test this "streaming" aspect (writing line by line), we can check calls to buffer.write.

    structure = Atoms("H", positions=[[0,0,0]], cell=[10,10,10])
    # Patch len to return large number
    # Patch get_positions to return a large array (mocked?)
    # Generating 1M lines of output in StringIO is fast enough for a unit test.

    # Actually, let's just use a generator-based test for `active_set.py` if relevant.
    # For `write_lammps_streaming`, let's verify it calls `write` many times.

    with patch("ase.Atoms.__len__", return_value=1000), \
         patch.object(Atoms, "get_positions", return_value=np.zeros((1000, 3))), \
         patch.object(Atoms, "get_chemical_symbols", return_value=["H"]*1000), \
         patch.object(Atoms, "get_cell", return_value=np.eye(3)):

         # Use a mock file object to count writes
         mock_file = MagicMock()
         write_lammps_streaming(mock_file, structure, ["H"])

         # Header writes + 1000 atom lines + mass lines + box lines
         # 1000 atoms -> 1000 calls for atoms
         assert mock_file.write.call_count > 1000
