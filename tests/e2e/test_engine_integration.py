import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult


@pytest.fixture
def mock_lammps_module() -> Any:
    """Mock lammps module for integration tests if not installed."""
    if "lammps" not in sys.modules:
        sys.modules["lammps"] = MagicMock()
    with patch("pyacemaker.interfaces.lammps_driver.lammps") as mock:
        # Configure mock to behave like lammps
        instance = mock.return_value
        # Mock extract_variable to return meaningful values
        instance.extract_variable.side_effect = lambda name, t, i: {
            "pe": -100.0,
            "step": 1000,
            "max_g": 0.05,
            "temp": 300.0,
            "halted": 0.0
        }.get(name, 0.0)
        # Mock get_natoms
        instance.get_natoms.return_value = 1
        # Mock gather_atoms
        # We need to ensure get_atoms works.
        # But since we mock the module, LammpsDriver will use the mock.
        yield mock


def test_engine_integration_workflow(tmp_path: Path, mock_md_config: MDConfig, mock_lammps_module: Any) -> None:
    """
    Verifies that the engine can be instantiated and run.
    This simulates the full workflow but mocks the actual LAMMPS C++ calls.
    """
    # 1. Setup
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Update config to use a safe potential for testing (e.g., LJ if we were running real)
    # But here we mock.

    # 2. Execution
    engine = LammpsEngine(mock_md_config)
    result = engine.run(atoms, potential_path)

    # 3. Verification
    assert isinstance(result, MDSimulationResult)
    assert result.energy == -100.0
    assert result.n_steps == 1000

    # Verify that run was called on the mock
    mock_lammps_module.return_value.command.assert_called()

    # Check if input script was "written" (passed to command)
    # The current implementation might pass the whole script or line by line.
    # We can check calls.
    calls = mock_lammps_module.return_value.command.call_args_list
    script_lines = [call.args[0] for call in calls]
    full_script = "\n".join(script_lines)

    assert "units metal" in full_script
    assert "atom_style atomic" in full_script
    assert "pair_style hybrid/overlay" in full_script


def test_engine_integration_lammps_failure(tmp_path: Path, mock_md_config: MDConfig, mock_lammps_module: Any) -> None:
    """Tests proper error handling when LAMMPS crashes."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Simulate LAMMPS command failure
    mock_lammps_module.return_value.command.side_effect = RuntimeError("LAMMPS Error")

    engine = LammpsEngine(mock_md_config)

    with pytest.raises(RuntimeError, match="LAMMPS execution failed"):
        engine.run(atoms, potential_path)
