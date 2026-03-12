from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import MDHaltInterrupt
from pyacemaker.domain_models.md import MDConfig


from typing import Any
def test_engine_catches_md_halt_interrupt(tmp_path: Path) -> None:
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True
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

        # Simulate the driver raising an exception because the underlying lammps python wrapper raised MDHaltInterrupt
        def mock_run_file(*args: Any, **kwargs: Any) -> None:
            raise MDHaltInterrupt(step=500, epicenter_indices=[1, 2])

        mock_driver.run_file.side_effect = mock_run_file

        # Extract variables should not fail if we intercept properly
        mock_driver.extract_variable.side_effect = lambda name: {
            "pe": -100.0,
            "temp": 300.0,
            "step": 500,
            "max_g": 0.6,
        }.get(name, 0.0)

        mock_driver.get_forces.return_value = __import__("numpy").array([[0.0, 0.0, 0.0]])
        mock_driver.get_stress.return_value = __import__("numpy").array([0.0]*6)

        result = engine.run(atoms, potential)

        assert result.halted is True
        assert result.halt_step == 500
        assert result.epicenter_indices == [1, 2]
