from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_lammps_engine_halt_step_populated(tmp_path: Path) -> None:
    from pyacemaker.domain_models.workflow import WorkflowConfig

    # Mock config
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path=str(tmp_path / "state.json"),
        data_dir=str(tmp_path / "data"),
        active_learning_dir=str(tmp_path / "al"),
        potentials_dir=str(tmp_path / "pots"),
        otf={"fix_halt": True},
    )

    from unittest.mock import patch

    with patch("pyacemaker.core.engine.LammpsDriver") as MockDriver:
        driver = MockDriver.return_value

        # Simulate halted run
        driver.extract_variable.side_effect = lambda name: {
            "pe": -100.0,
            "step": 500,  # Halted at 500
            "max_g": 10.0,
            "temp": 300.0,
        }.get(name, 0.0)

        driver.get_forces.return_value = np.zeros((1, 3))
        driver.get_stress.return_value = np.zeros(6)

        engine = LammpsEngine(config, workflow_config)

        with patch.object(engine.parser, "_evaluate_uncertainty_stream", return_value=([0], True)):
            atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
            pot_path = tmp_path / "pot.yace"
            pot_path.touch()

            result = engine.run(atoms, pot_path)

        assert result.halted is True
        assert result.halt_step == 500
        assert result.halt_structure_path is not None


def test_lammps_engine_halt_step_none_if_not_halted(tmp_path: Path) -> None:
    from pyacemaker.domain_models.workflow import WorkflowConfig

    # Mock config
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path=str(tmp_path / "state.json"),
        data_dir=str(tmp_path / "data"),
        active_learning_dir=str(tmp_path / "al"),
        potentials_dir=str(tmp_path / "pots"),
        otf={"fix_halt": True},
    )

    with patch("pyacemaker.core.engine.LammpsDriver") as MockDriver:
        driver = MockDriver.return_value

        # Simulate complete run
        driver.extract_variable.side_effect = lambda name: {
            "pe": -100.0,
            "step": 1000,  # Completed
            "max_g": 0.05,
            "temp": 300.0,
        }.get(name, 0.0)

        driver.get_forces.return_value = np.zeros((1, 3))
        driver.get_stress.return_value = np.zeros(6)

        engine = LammpsEngine(config, workflow_config)
        atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
        pot_path = tmp_path / "pot.yace"
        pot_path.touch()

        result = engine.run(atoms, pot_path)

        assert result.halted is False
        assert result.halt_step is None
        assert result.halt_structure_path is None
