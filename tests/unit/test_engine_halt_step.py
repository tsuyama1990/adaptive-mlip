from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_lammps_engine_halt_step_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Mock config
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True,
    )

    with patch("pyacemaker.core.engine.LammpsDriver") as MockDriver:
        driver = MockDriver.return_value

        # Simulate halted run at step 500
        driver.extract_variable.side_effect = lambda name: {
            "pe": -100.0,
            "step": 500, # Halted at 500
            "max_g": 10.0,
            "temp": 300.0
        }.get(name, 0.0)

        engine = LammpsEngine(config)
        atoms = Atoms("H", cell=[10,10,10], pbc=True)
        pot_path = tmp_path / "pot.yace"
        pot_path.touch()

        result = engine.run(atoms, pot_path)

        assert result.halted is True
        assert result.halt_step == 500

def test_lammps_engine_halt_step_none_if_not_halted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Mock config
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True,
    )

    with patch("pyacemaker.core.engine.LammpsDriver") as MockDriver:
        driver = MockDriver.return_value

        # Simulate complete run
        driver.extract_variable.side_effect = lambda name: {
            "pe": -100.0,
            "step": 1000, # Completed
            "max_g": 0.05,
            "temp": 300.0
        }.get(name, 0.0)

        engine = LammpsEngine(config)
        atoms = Atoms("H", cell=[10,10,10], pbc=True)
        pot_path = tmp_path / "pot.yace"
        pot_path.touch()

        result = engine.run(atoms, pot_path)

        assert result.halted is False
        assert result.halt_step is None
