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

    # Verify that run was called on the mock (run_file calls lmp.file)
    mock_lammps_module.return_value.file.assert_called()

    # Check if input script was "written" to file
    # We can check the file argument passed to file()
    file_call = mock_lammps_module.return_value.file.call_args
    script_path = Path(file_call[0][0])

    # Read the script file (it should still exist or we mock reading it if we care,
    # but here we just check integration flow. The file writing logic is tested in unit tests)
    # Actually temp dir might be gone?
    # LammpsFileManager uses tempfile.TemporaryDirectory. It cleans up on exit of context.
    # But run() calls prepare_workspace then execute inside ctx.
    # Engine.run returns result after ctx exit?
    # No, Engine.run does: with ctx: execute(); return result.
    # So ctx exits before return. File is gone.
    # We can't read script content here easily unless we mock open or check calls before exit.

    # But we can assume if file() was called, script generation happened.
    # Unit tests cover generator content.


def test_engine_integration_lammps_failure(tmp_path: Path, mock_md_config: MDConfig, mock_lammps_module: Any) -> None:
    """Tests proper error handling when LAMMPS crashes."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Simulate LAMMPS command failure
    mock_lammps_module.return_value.file.side_effect = RuntimeError("LAMMPS Error")

    engine = LammpsEngine(mock_md_config)

    with pytest.raises(RuntimeError, match="LAMMPS engine execution failed"):
        engine.run(atoms, potential_path)
