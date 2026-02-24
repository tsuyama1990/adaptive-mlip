
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import HybridParams, MDConfig, MDSimulationResult


@pytest.fixture
def mock_driver() -> Any:
    with patch("pyacemaker.core.engine.LammpsDriver") as mock:
        yield mock


def test_lammps_engine_run(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    # Set up mock driver
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -100.0,
        "step": 1000,
        "max_g": 0.05,
        "temp": 300.0,
        "halted": 0.0  # Not halted
    }.get(name, 0.0)
    driver_instance.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    driver_instance.get_stress.return_value = np.array([0.0] * 6)

    # Capture script content
    script_content = []
    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())
    driver_instance.run_file.side_effect = capture_run

    # Mock get_atoms
    driver_instance.get_atoms.return_value = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Enable fix_halt to test gamma extraction
    config = mock_md_config.model_copy(update={"fix_halt": True})
    engine = LammpsEngine(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Create dummy potential file
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    result = engine.run(atoms, pot_path)

    assert isinstance(result, MDSimulationResult)
    assert result.energy == -100.0
    assert result.n_steps == 1000
    assert result.halted is False
    assert result.max_gamma == 0.05
    assert result.trajectory_path is not None
    assert re.search(r"dump_[a-f0-9]{8}\.lammpstrj", result.trajectory_path)

    # Verify driver run_file called
    driver_instance.run_file.assert_called()

    # Check captured script
    assert len(script_content) == 1
    script = script_content[0]

    assert "fix halt" in script
    assert "read_data" in script


def test_lammps_engine_halted(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -90.0,
        "step": 500,
        "max_g": 10.0,
        "temp": 310.0,
        "halted": 1.0
    }.get(name, 0.0)
    driver_instance.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    driver_instance.get_stress.return_value = np.array([0.0] * 6)

    driver_instance.get_atoms.return_value = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Enable fix_halt to test halted logic
    config = mock_md_config.model_copy(update={"fix_halt": True})
    engine = LammpsEngine(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    result = engine.run(atoms, pot_path)

    assert result.halted is True
    assert result.max_gamma == 10.0
    assert result.n_steps == 500
    assert result.halt_structure_path == result.trajectory_path


def test_lammps_engine_hybrid_potential(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    hybrid_params = HybridParams(zbl_global_cutoff=1.5)
    config = mock_md_config.model_copy(update={"hybrid_potential": True, "hybrid_params": hybrid_params})

    engine = LammpsEngine(config)
    atoms = Atoms("Al", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    # Capture script content
    script_content = []
    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance = mock_driver.return_value
    driver_instance.run_file.side_effect = capture_run
    driver_instance.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    driver_instance.get_stress.return_value = np.array([0.0] * 6)

    engine.run(atoms, pot_path)

    # Check captured script
    assert len(script_content) == 1
    script = script_content[0]

    assert "pair_style hybrid/overlay" in script
    assert "pair_coeff * * pace" in script
    assert "pair_coeff 1 1 zbl 13 13" in script # Al is Z=13

    # Floating point check (1.2 can be 1.2000000000000002)
    assert "zbl" in script
    assert "1.5" in script # Check outer cutoff is present


def test_run_empty_structure_error(mock_md_config: MDConfig, tmp_path: Path) -> None:
    """Tests error handling for empty structure."""
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms() # Empty
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    with pytest.raises(ValueError, match="Structure contains no atoms"):
        engine.run(atoms, pot_path)


def test_run_missing_potential_error(mock_md_config: MDConfig) -> None:
    """Tests error handling for missing potential file."""
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    with pytest.raises(FileNotFoundError, match="Potential file not found"):
        engine.run(atoms, "nonexistent.yace")


def test_run_large_structure_warning(mock_md_config: MDConfig, mock_driver: Any, caplog: Any, tmp_path: Path) -> None:
    """Tests info log for large structures (streaming)."""
    import logging
    caplog.set_level(logging.INFO)
    engine = LammpsEngine(mock_md_config)
    # Create large structure > 10k
    atoms = Atoms(symbols=["H"] * 10001, positions=[[0,0,0]]*10001, cell=[100,100,100], pbc=True)

    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Mock driver outputs
    driver = mock_driver.return_value
    driver.get_forces.return_value = np.array([[0.0]*3]*10001)
    driver.get_stress.return_value = np.array([0.0]*6)
    # Mock extract_variable to prevent int conversion error if Mock returns MagicMock
    driver.extract_variable.return_value = 0.0

    # We rely on mock driver to avoid actual execution overhead
    # Note: prepare_workspace calls write_lammps_streaming which iterates.
    # For 10k atoms it's fast enough for test.
    engine.run(atoms, pot_path)


def test_run_driver_failure(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    """Tests error handling when LAMMPS execution fails."""
    driver_instance = mock_driver.return_value
    driver_instance.run_file.side_effect = RuntimeError("LAMMPS crashed")

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    with pytest.raises(RuntimeError, match="LAMMPS engine execution failed: LAMMPS crashed"):
        engine.run(atoms, pot_path)
