from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import EngineError
from pyacemaker.domain_models.md import MDConfig


@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
    )


@pytest.fixture
def mock_driver(mocker: Any) -> Any:
    # Patch LammpsDriver where it is imported in engine.py
    return mocker.patch("pyacemaker.core.engine.LammpsDriver")


@pytest.fixture
def mock_file_manager(mocker: Any) -> Any:
    fm = mocker.patch("pyacemaker.core.engine.LammpsFileManager")
    instance = fm.return_value
    # Mock prepare_workspace to return a context manager
    instance.prepare_workspace.return_value = (
        MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        MagicMock(parent=MagicMock()),
        MagicMock(),
        MagicMock(),
        ["H"],
    )
    return instance


def test_run_success(
    mock_md_config: MDConfig, mock_driver: Any, mock_file_manager: Any, tmp_path: Path
) -> None:
    """Tests successful run."""
    # Setup driver mock
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -10.0,
        "temp": 300.0,
        "step": 1000,
    }.get(name, 0.0)

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    result = engine.run(atoms, pot_path)

    assert result.energy == -10.0
    assert result.temperature == 300.0
    assert not result.halted


def test_run_driver_failure(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    """Tests error handling when LAMMPS execution fails."""
    driver_instance = mock_driver.return_value
    driver_instance.run.side_effect = RuntimeError("LAMMPS crashed")

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Engine should wrap RuntimeError in EngineError
    with pytest.raises(EngineError, match="LAMMPS execution failed"):
        engine.run(atoms, pot_path)
