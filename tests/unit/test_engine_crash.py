import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_engine_catches_lammps_crash(tmp_path: Path) -> None:
    """Simulate catastrophic LAMMPS crash as required by Cycle 04."""
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000
    )
    engine = LammpsEngine(config)

    atoms = Atoms("Al", positions=[[0, 0, 0]])
    potential = "dummy.yace"

    with patch("pyacemaker.core.engine.LammpsDriver") as mock_driver_class, \
         patch("pyacemaker.core.engine.LammpsScriptGenerator"), \
         patch.object(engine, "_prepare_simulation_env") as mock_prep:

        mock_ctx = MagicMock()
        mock_ctx.name = str(tmp_path)
        mock_prep.return_value = (mock_ctx, tmp_path / "data.lmp", tmp_path / "dump.lammpstrj", tmp_path / "log.lammps", ["Al"], Path("dummy.yace"))

        mock_driver = mock_driver_class.return_value

        def mock_run_file(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(subprocess.CalledProcessError(1, "cmd", stderr="Simulated LAMMPS Crash"))

        mock_driver.run_file.side_effect = mock_run_file

        with pytest.raises(RuntimeError, match="Simulation execution failed"):
            engine.run(atoms, potential)
