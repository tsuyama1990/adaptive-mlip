import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.training import TrainingConfig


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


def test_training_config_filename_validation() -> None:
    # Valid
    TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=1,
        output_filename="valid.yace"
    )

    # Invalid
    with pytest.raises(ValidationError):
        TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=1,
            output_filename="path/traversal.yace"
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
