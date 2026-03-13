import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.md import (
    MCConfig,
    MDConfig,
    MDRampingConfig,
    MDSimulationResult,
    ZBLConfig,
)


def test_zbl_config_valid() -> None:
    """Tests valid ZBLConfig."""
    params = ZBLConfig(zbl_cut_inner=1.5, zbl_cut_outer=2.0)
    assert params.zbl_cut_inner == 1.5
    assert params.zbl_cut_outer == 2.0


def test_zbl_config_invalid() -> None:
    """Tests invalid ZBLConfig (negative cutoff)."""
    with pytest.raises(ValidationError):
        ZBLConfig(zbl_cut_inner=-1.0)


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
    assert config.zbl.zbl_cut_inner > 0
    assert config.dump_freq == 100
    assert config.thermo_freq == 10


def test_md_config_with_hybrid_params() -> None:
    """Tests MDConfig with custom ZBL Cutoffs."""
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        zbl=ZBLConfig(zbl_cut_inner=1.0, zbl_cut_outer=1.5),
    )
    assert config.zbl.zbl_cut_inner == 1.0


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
        log_path="log.lammps",
    )
    assert result.energy == -500.0
    assert result.trajectory_path == "dump.lammpstrj"
    assert result.log_path == "log.lammps"
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


def test_md_ramping_config() -> None:
    """Tests MDRampingConfig validation."""
    # Valid
    config = MDRampingConfig(temp_start=300.0, temp_end=1000.0, press_start=1.0, press_end=100.0)
    assert config.temp_start == 300.0
    assert config.press_end == 100.0

    # Test invalid temperature
    with pytest.raises(ValidationError):
        MDRampingConfig(temp_start=-50.0)


def test_mc_config() -> None:
    """Tests MCConfig validation."""
    # Valid
    config = MCConfig(swap_freq=100, swap_prob=0.5, seed=123)
    assert config.swap_freq == 100
    assert config.swap_prob == 0.5

    # Test invalid probability (prob > 1.0)
    with pytest.raises(ValidationError):
        MCConfig(swap_freq=10, swap_prob=1.5)

    # Test invalid frequency (freq <= 0)
    with pytest.raises(ValidationError):
        MCConfig(swap_freq=0, swap_prob=0.5)


def test_md_config_with_ramping_and_mc() -> None:
    """Tests MDConfig integration with Ramping and MC."""
    ramping = MDRampingConfig(temp_start=100.0, temp_end=500.0)
    mc = MCConfig(swap_freq=50, swap_prob=0.1, seed=123)

    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        ramping=ramping,
        mc=mc,
    )

    assert config.ramping is not None
    assert config.ramping.temp_start == 100.0
    assert config.mc is not None
    assert config.mc.swap_freq == 50
