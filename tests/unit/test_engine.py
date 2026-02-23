from typing import Any
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult


@pytest.fixture
def mock_driver() -> Any:
    with patch("pyacemaker.core.engine.LammpsDriver") as mock:
        yield mock


def test_lammps_engine_run(mock_md_config: MDConfig, mock_driver: Any) -> None:
    # Set up mock driver
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -100.0,
        "step": 1000,
        "max_g": 0.05,
        "temp": 300.0,
        "halted": 0.0  # Not halted
    }.get(name, 0.0)

    # Mock get_atoms to return input structure
    driver_instance.get_atoms.return_value = Atoms("H", cell=[10, 10, 10], pbc=True)

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    result = engine.run(atoms, "potential.yace")

    assert isinstance(result, MDSimulationResult)
    assert result.energy == -100.0
    assert result.n_steps == 1000
    assert result.halted is False
    assert result.max_gamma == 0.05
    assert result.trajectory_path.startswith("dump_")
    assert result.trajectory_path.endswith(".lammpstrj")

    # Verify driver run called with script
    driver_instance.run.assert_called()
    script = driver_instance.run.call_args[0][0]
    assert "pair_style" in script
    assert "fix halt" in script


def test_lammps_engine_halted(mock_md_config: MDConfig, mock_driver: Any) -> None:
    # Set up mock driver to simulate halt
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -90.0,
        "step": 500,
        "max_g": 10.0,  # High gamma
        "temp": 310.0,
        "halted": 1.0  # Halted
    }.get(name, 0.0)

    driver_instance.get_atoms.return_value = Atoms("H", cell=[10, 10, 10], pbc=True)

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    result = engine.run(atoms, "potential.yace")

    assert result.halted is True
    assert result.max_gamma == 10.0
    assert result.n_steps == 500


def test_lammps_engine_hybrid_potential(mock_md_config: MDConfig, mock_driver: Any) -> None:
    # Enable hybrid potential
    config = mock_md_config.model_copy(update={"hybrid_potential": True, "hybrid_params": {"zbl_cut_inner": 1.0}})

    engine = LammpsEngine(config)
    atoms = Atoms("Al", cell=[10, 10, 10], pbc=True)
    engine.run(atoms, "potential.yace")

    driver_instance = mock_driver.return_value
    script = driver_instance.run.call_args[0][0]

    assert "pair_style hybrid/overlay" in script
    assert "pair_coeff * * pace" in script
    # Al is Z=13. Type 1.
    # pair_coeff 1 1 zbl 13 13
    assert "pair_coeff 1 1 zbl 13 13" in script
