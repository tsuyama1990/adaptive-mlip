
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ase import Atoms

from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.domain_models.md import MDConfig


def test_prepare_workspace(tmp_path: Path, mock_md_config: MDConfig) -> None:
    # Use tmp_path for temp_dir to avoid /dev/shm in tests
    # Note: MDConfig does not have temp_dir field, but LammpsFileManager uses default.
    # We can't set temp_dir on MDConfig.
    # We can patch tempfile.TemporaryDirectory to use tmp_path

    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = str(tmp_path)
        mock_temp.return_value.name = str(tmp_path)

        manager = LammpsFileManager(mock_md_config)
        atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

        ctx, data, dump, log, els = manager.prepare_workspace(atoms)

        # No context manager entry needed if we mocked it, but prepare_workspace returns the object
        # which we can use in 'with'.
        # However, our mock setup for return_value is for the constructor call.
        # prepare_workspace calls TemporaryDirectory().

        # Actually, let's just let it use default temp dir, but verify data file creation.
        # But we need to verify where it is.

    # Real implementation test
    manager = LammpsFileManager(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    ctx, data, dump, log, els = manager.prepare_workspace(atoms)

    with ctx:
        assert data.exists()
        assert dump.parent == Path.cwd()
        assert log.parent == Path.cwd()
        assert els == ["H"]


def test_prepare_workspace_large_structure_warning(mock_md_config: MDConfig, caplog: Any) -> None:
    import logging
    caplog.set_level(logging.INFO)
    manager = LammpsFileManager(mock_md_config)
    atoms = Atoms(symbols=["H"] * 10001, positions=[[0,0,0]]*10001, cell=[100,100,100], pbc=True)

    # We patch write_lammps_streaming to avoid actual I/O for large structure test
    with patch("pyacemaker.core.io_manager.write_lammps_streaming") as mock_stream:
        ctx, _, _, _, _ = manager.prepare_workspace(atoms)
        with ctx:
            pass
        mock_stream.assert_called_once()
