from pathlib import Path
from typing import Any

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import EngineError
from pyacemaker.domain_models import MDConfig


@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
    )


@pytest.fixture
def mock_lammps_module(mocker: Any) -> Any:
    # Patch the lammps module where it is imported in lammps_driver.py
    # Since LammpsDriver imports `from lammps import lammps`
    # We need to patch `pyacemaker.interfaces.lammps_driver.lammps`
    return mocker.patch("pyacemaker.interfaces.lammps_driver.lammps")


def test_engine_integration_success(
    tmp_path: Path, mock_md_config: MDConfig, mock_lammps_module: Any
) -> None:
    """Tests successful execution flow."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Configure mock
    instance = mock_lammps_module.return_value
    instance.extract_variable.side_effect = lambda name, group, type_: (
        100 if name == "step" else -5.0
    )

    engine = LammpsEngine(mock_md_config)
    result = engine.run(atoms, potential_path)

    assert result.energy == -5.0
    assert not result.halted
    assert instance.command.called


def test_engine_integration_lammps_failure(
    tmp_path: Path, mock_md_config: MDConfig, mock_lammps_module: Any
) -> None:
    """Tests proper error handling when LAMMPS crashes."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Simulate LAMMPS command failure
    mock_lammps_module.return_value.command.side_effect = RuntimeError("LAMMPS Error")

    engine = LammpsEngine(mock_md_config)

    # The Engine now wraps RuntimeError in EngineError
    with pytest.raises(
        EngineError, match="Engine execution failed: LAMMPS execution failed: LAMMPS Error"
    ):
        engine.run(atoms, potential_path)
