import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.md import HybridParams, MDConfig, MDSimulationResult


def test_hybrid_params_valid() -> None:
    """Tests valid HybridParams."""
    params = HybridParams(zbl_global_cutoff=2.0)
    assert params.zbl_global_cutoff == 2.0


def test_hybrid_params_invalid() -> None:
    """Tests invalid HybridParams (extra fields)."""
    with pytest.raises(ValidationError):
        HybridParams(zbl_cut_inner=1.0)  # type: ignore[call-arg]


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
    assert isinstance(config.hybrid_params, HybridParams)
    assert config.dump_freq == 100
    assert config.thermo_freq == 10


def test_md_config_with_hybrid_params() -> None:
    """Tests MDConfig with custom HybridParams."""
    hybrid_params = HybridParams(zbl_global_cutoff=1.5)
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        hybrid_params=hybrid_params,
    )
    assert config.hybrid_params.zbl_global_cutoff == 1.5


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
        stress=[0.0] * 6,
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
        stress=[0.0] * 6,
        halted=True,
        max_gamma=10.0,
        n_steps=50,
        temperature=310.0,
        trajectory_path="dump.lammpstrj",
        log_path="log.lammps",
        halt_structure_path="halt_structure.xyz",
    )
    assert result.halted is True
    assert result.max_gamma == 10.0
    assert result.halt_structure_path == "halt_structure.xyz"
