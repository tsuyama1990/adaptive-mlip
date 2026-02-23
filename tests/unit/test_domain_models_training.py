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
