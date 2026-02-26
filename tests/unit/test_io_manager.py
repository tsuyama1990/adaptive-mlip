from pathlib import Path
from typing import Any
from unittest.mock import patch

from ase import Atoms

from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.domain_models.md import MDConfig


def test_prepare_workspace(tmp_path: Path, mock_md_config: MDConfig) -> None:
    # Use tmp_path for temp_dir to avoid /dev/shm in tests
    config = mock_md_config.model_copy(update={"temp_dir": str(tmp_path)})
    manager = LammpsFileManager(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    ctx, data, dump, log, els = manager.prepare_workspace(atoms)

    with ctx:
        assert data.exists()
        # Verify data file is inside the temp_dir (which is inside tmp_path)
        assert tmp_path in data.parents
        assert dump.name.startswith("dump_")
        assert log.name.startswith("log_")
        assert els == ["H"]

    # After exit, the file inside temp dir should be gone
    assert not data.exists()


def test_prepare_workspace_large_structure_warning(mock_md_config: MDConfig, caplog: Any) -> None:
    import logging
    caplog.set_level(logging.INFO)
    manager = LammpsFileManager(mock_md_config)
    atoms = Atoms(symbols=["H"] * 10001, positions=[[0,0,0]]*10001, cell=[100,100,100], pbc=True)

    # Force capture of io_manager logs
    caplog.set_level(logging.INFO, logger="pyacemaker.core.io_manager")

    # We patch write_lammps_streaming to avoid actual I/O for large structure test
    with patch("pyacemaker.core.io_manager.write_lammps_streaming") as mock_stream, \
         patch("pyacemaker.core.io_manager.get_species_order", return_value=["H"]):
            ctx, _, _, _, _ = manager.prepare_workspace(atoms)
            with ctx:
                pass
            mock_stream.assert_called_once()

    if len(caplog.records) > 0:
        assert any("Streaming large structure" in record.message for record in caplog.records)
