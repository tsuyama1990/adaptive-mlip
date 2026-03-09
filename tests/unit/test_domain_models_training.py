import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.training import PacemakerConfig, TrainingConfig


def test_training_config_defaults() -> None:
    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=1,
    )
    assert config.delta_learning is False
    assert config.active_set_optimization is False
    assert config.active_set_size is None
    assert config.seed == 42
    assert config.max_iterations == 1000
    assert config.batch_size == 10

    # Check Pacemaker config defaults
    assert config.pacemaker.embedding_type == "FinnisSinclair"
    assert config.pacemaker.fs_parameters == [1.0, 1.0, 1.0, 1.5]
    assert config.pacemaker.optimizer == "BFGS"


def test_training_config_filename_validation() -> None:
    # Valid
    TrainingConfig(
        potential_type="ace", cutoff_radius=5.0, max_basis_size=1, output_filename="valid.yace"
    )

    # Invalid
    with pytest.raises(ValidationError):
        TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=1,
            output_filename="path/traversal.yace",
        )


def test_training_config_active_set_size_valid() -> None:
    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=1,
        active_set_optimization=True,
        active_set_size=10,
    )
    assert config.active_set_size == 10


def test_training_config_active_set_size_invalid_zero() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=1,
            active_set_size=0,
        )


def test_training_config_active_set_size_invalid_negative() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=1,
            active_set_size=-1,
        )


def test_training_config_active_set_required() -> None:
    with pytest.raises(ValidationError, match="active_set_size must be set"):
        TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=1,
            active_set_optimization=True,
            # active_set_size missing
        )


def test_pacemaker_config_custom_values() -> None:
    pm_config = PacemakerConfig(
        embedding_type="Custom",
        fs_parameters=[0.5],
        ndensity=3,
        rad_base="Bessel",
        rad_parameters=[2.0],
        max_deg=8,
        r0=2.0,
        loss_kappa=0.5,
        loss_l1_coeffs=1e-5,
        loss_l2_coeffs=1e-5,
        repulsion_sigma=0.1,
        optimizer="Adam",
    )

    config = TrainingConfig(
        potential_type="ace", cutoff_radius=5.0, max_basis_size=1, pacemaker=pm_config
    )

    assert config.pacemaker.optimizer == "Adam"
    assert config.pacemaker.max_deg == 8
