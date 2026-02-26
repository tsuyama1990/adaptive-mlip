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

    # 2. Execution
    engine = LammpsEngine(mock_md_config)
    result = engine.run(atoms, potential_path)

    # 3. Verification
    assert isinstance(result, MDSimulationResult)
    assert result.energy == -100.0
    assert result.n_steps == 1000

    # Verify that run was called on the mock (run_file calls lmp.command now)
    mock_lammps_module.return_value.command.assert_called()

    # We can verify the content by inspecting calls to command()
    calls = mock_lammps_module.return_value.command.call_args_list
    # The script execution iterates over lines and calls command() for each valid line.
    assert any("units metal" in c[0][0] for c in calls)
    assert any("pair_style hybrid/overlay" in c[0][0] for c in calls)


def test_engine_integration_lammps_failure(tmp_path: Path, mock_md_config: MDConfig, mock_lammps_module: Any) -> None:
    """Tests proper error handling when LAMMPS crashes."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Simulate LAMMPS command failure (on command(), not file())
    mock_lammps_module.return_value.command.side_effect = RuntimeError("LAMMPS Error")

    engine = LammpsEngine(mock_md_config)

    # Updated match string
    with pytest.raises(RuntimeError, match="Simulation execution failed"):
        engine.run(atoms, potential_path)
