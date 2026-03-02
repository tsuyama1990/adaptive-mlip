import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.workflow import (
    ActiveLearningThresholds,
    CutoutConfig,
    DistillationConfig,
    LoopStrategyConfig,
)


def test_active_learning_thresholds_valid():
    t = ActiveLearningThresholds(threshold_call_dft=1.0, threshold_add_train=2.0, smooth_steps=3)
    assert t.threshold_call_dft == 1.0
    assert t.threshold_add_train == 2.0
    assert t.smooth_steps == 3

def test_active_learning_thresholds_invalid():
    with pytest.raises(ValidationError):
        ActiveLearningThresholds(threshold_call_dft="invalid", threshold_add_train=2.0, smooth_steps=3)
    with pytest.raises(ValidationError):
        ActiveLearningThresholds(threshold_call_dft=1.0, threshold_add_train=2.0, smooth_steps=-1)

def test_cutout_config_valid():
    c = CutoutConfig(core_radius=3.0, buffer_radius=5.0, enable_pre_relaxation=True, enable_passivation=True)
    assert c.core_radius == 3.0
    assert c.buffer_radius == 5.0
    assert c.enable_pre_relaxation is True
    assert c.enable_passivation is True

def test_cutout_config_invalid():
    with pytest.raises(ValidationError):
        CutoutConfig(core_radius="invalid", buffer_radius=5.0, enable_pre_relaxation=True, enable_passivation=True)

def test_distillation_config_valid():
    d = DistillationConfig(enable=True, mace_model_path="model.pt", uncertainty_threshold=0.5, sampling_counts=1000)
    assert d.enable is True
    assert d.mace_model_path == "model.pt"
    assert d.uncertainty_threshold == 0.5
    assert d.sampling_counts == 1000

def test_distillation_config_invalid():
    with pytest.raises(ValidationError):
        DistillationConfig(enable=True, mace_model_path="model.pt", uncertainty_threshold=0.5, sampling_counts=-10)

def test_loop_strategy_config_valid():
    s = LoopStrategyConfig(use_tiered_oracle=True, incremental_update=True, replay_buffer_size=1000, baseline_potential_type="LJ")
    assert s.use_tiered_oracle is True
    assert s.incremental_update is True
    assert s.replay_buffer_size == 1000
    assert s.baseline_potential_type == "LJ"

def test_loop_strategy_config_invalid():
    with pytest.raises(ValidationError):
        LoopStrategyConfig(use_tiered_oracle=True, incremental_update=True, replay_buffer_size=-1000, baseline_potential_type="LJ")
