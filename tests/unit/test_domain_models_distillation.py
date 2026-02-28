import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.distillation import (
    ActiveLearningThresholds,
    CutoutConfig,
    DistillationConfig,
    LoopStrategyConfig,
)


def test_distillation_config_defaults():
    # Test DistillationConfig defaults
    d_config = DistillationConfig()
    assert d_config.enable is True
    assert d_config.mace_model_path == "mace-mp-0-medium"
    assert d_config.uncertainty_threshold == 0.05
    assert d_config.sampling_structures_per_system == 1000

    # Test ActiveLearningThresholds defaults
    al_config = ActiveLearningThresholds()
    assert al_config.threshold_call_dft == 0.05
    assert al_config.threshold_add_train == 0.02
    assert al_config.smooth_steps == 3

    # Test CutoutConfig defaults
    c_config = CutoutConfig()
    assert c_config.core_radius == 4.0
    assert c_config.buffer_radius == 3.0
    assert c_config.enable_pre_relaxation is True
    assert c_config.enable_passivation is True
    assert c_config.passivation_element == "H"

    # Test LoopStrategyConfig defaults
    l_config = LoopStrategyConfig()
    assert l_config.use_tiered_oracle is True
    assert l_config.incremental_update is True
    assert l_config.replay_buffer_size == 500
    assert l_config.baseline_potential_type == "LJ"
    assert isinstance(l_config.thresholds, ActiveLearningThresholds)


def test_cutout_config_invalid_radii():
    with pytest.raises(ValidationError):
        CutoutConfig(core_radius=-1.0)

    with pytest.raises(ValidationError):
        CutoutConfig(buffer_radius=-1.0)


def test_distillation_config_invalid_values():
    with pytest.raises(ValidationError):
        DistillationConfig(uncertainty_threshold=-0.1)

    with pytest.raises(ValidationError):
        DistillationConfig(sampling_structures_per_system=0)


def test_active_learning_invalid_values():
    with pytest.raises(ValidationError):
        ActiveLearningThresholds(smooth_steps=0)


def test_cutout_config_invalid_element():
    with pytest.raises(ValidationError):
        CutoutConfig(passivation_element="")
