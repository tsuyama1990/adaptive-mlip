
import re
from typing import Any
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import HybridParams, MDConfig, MDSimulationResult


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
    # Verify trajectory path format (dump_{uuid}.lammpstrj)
    assert result.trajectory_path is not None
    assert re.search(r"dump_[a-f0-9]{8}\.lammpstrj", result.trajectory_path)

    # Verify driver run called with script
    driver_instance.run.assert_called()
    script = driver_instance.run.call_args[0][0]
    assert "pair_style" in script
    assert "fix halt" in script
    # Verify that data file path in script is absolute (from temp dir)
    # read_data /tmp/.../data_...
    match = re.search(r"read_data (.*)", script)
    assert match
    data_file = match.group(1)
    assert "data_" in data_file
    assert ".lmp" in data_file


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
    assert result.halt_structure_path == result.trajectory_path


def test_lammps_engine_hybrid_potential(mock_md_config: MDConfig, mock_driver: Any) -> None:
    # Enable hybrid potential
    hybrid_params = HybridParams(zbl_cut_inner=1.0, zbl_cut_outer=1.5)
    config = mock_md_config.model_copy(update={"hybrid_potential": True, "hybrid_params": hybrid_params})

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

    # Check custom cutoffs
    assert "1.0 1.5" in script # zbl 1.0 1.5


def test_generate_input_script(mock_md_config: MDConfig) -> None:
    """Tests the input script generation logic explicitly."""
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    script = engine._generate_input_script(
        atoms, "pot.yace", "data.lmp", "dump.lmp", ["H"]
    )

    assert "pair_style hybrid/overlay" in script
    assert "pair_coeff * * pace pot.yace H" in script
    assert "pair_coeff 1 1 zbl 1 1" in script # H is Z=1
    assert "compute gamma all pace" in script
    assert "fix halt_check all halt" in script
    assert "fix npt all npt" in script
    # Check damping
    # timestep 0.001 -> tdamp 0.1, pdamp 1.0
    assert "temp 300.0 300.0 0.1" in script
    assert "iso 1.0 1.0 1.0" in script
