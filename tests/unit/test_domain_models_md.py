import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.md import MDConfig, MDSimulationResult


def test_md_config_valid() -> None:
    """Tests valid MD configuration."""
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        dump_freq=100,
        thermo_freq=10,
        hybrid_potential=True,
        uncertainty_threshold=0.1,
    )
    assert config.temperature == 300.0
    assert config.hybrid_potential is True
    assert config.dump_freq == 100
    assert config.thermo_freq == 10


def test_md_config_invalid_temperature() -> None:
    """Tests invalid temperature (negative)."""
    with pytest.raises(ValidationError):
        MDConfig(
            temperature=-100.0,
            pressure=1.0,
            timestep=0.001,
            n_steps=1000,
        )


def test_md_config_invalid_steps() -> None:
    """Tests invalid steps (negative)."""
    with pytest.raises(ValidationError):
        MDConfig(
            temperature=300.0,
            pressure=1.0,
            timestep=0.001,
            n_steps=-10,
        )


def test_md_simulation_result_valid() -> None:
    """Tests valid MDSimulationResult."""
    result = MDSimulationResult(
        energy=-500.0,
        forces=[[0.0, 0.0, 0.0]],
        halted=False,
        max_gamma=0.05,
        n_steps=1000,
        temperature=300.0,
        trajectory_path="dump.lammpstrj",
    )
    assert result.energy == -500.0
    assert result.trajectory_path == "dump.lammpstrj"
    assert result.halt_structure_path is None


def test_md_simulation_result_halted() -> None:
    """Tests halted MDSimulationResult."""
    result = MDSimulationResult(
        energy=-400.0,
        forces=[[0.0, 0.0, 0.0]],
        halted=True,
        max_gamma=10.0,
        n_steps=50,
        temperature=310.0,
        halt_structure_path="halt_structure.xyz",
    )
    assert result.halted is True
    assert result.max_gamma == 10.0
    assert result.halt_structure_path == "halt_structure.xyz"
