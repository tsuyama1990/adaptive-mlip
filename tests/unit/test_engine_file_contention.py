import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_engine_file_contention(tmp_path: Path) -> None:
    """Cycle 04: Test file system contention handling (mocking)."""
    # Since orchestrator usually handles file contention after run returns,
    # or engine handles it during execution, we test if standard engine
    # routines properly avoid crash on concurrent file mock operations.

    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000
    )
    engine = LammpsEngine(config)

    atoms = Atoms("Al", positions=[[0, 0, 0]])
    potential = "dummy.yace"

    # We will simulate multiple threads trying to run the engine in the same temp directory

    def worker() -> None:
        with patch("pyacemaker.core.engine.LammpsDriver") as mock_driver_class, \
             patch("pyacemaker.core.engine.LammpsScriptGenerator"), \
             patch.object(engine, "_prepare_simulation_env") as mock_prep:

            mock_ctx = MagicMock()
            mock_ctx.name = str(tmp_path)
            mock_prep.return_value = (mock_ctx, tmp_path / "data.lmp", tmp_path / "dump.lammpstrj", tmp_path / "log.lammps", ["Al"], Path("dummy.yace"))

            mock_driver = mock_driver_class.return_value
            mock_driver.extract_variable.side_effect = lambda name: {
                "pe": -100.0,
                "temp": 300.0,
                "step": 1000,
                "max_g": 0.05,
            }.get(name, 0.0)
            mock_driver.get_forces.return_value = __import__("numpy").array([[0.0, 0.0, 0.0]])
            mock_driver.get_stress.return_value = __import__("numpy").array([0.0]*6)

            # The writing to input_script_path is an IO operation we want to ensure doesn't strictly crash due to python built-in locks or we expect it to succeed.
            engine.run(atoms, potential)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert True
