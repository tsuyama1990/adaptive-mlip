
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.orchestrator import Orchestrator


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
