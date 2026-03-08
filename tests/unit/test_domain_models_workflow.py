import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.workflow import (
    ActiveLearningThresholds,
    CutoutConfig,
    DistillationConfig,
    LoopStrategyConfig,
    WorkflowConfig,
)


def test_distillation_config_defaults() -> None:
    config = DistillationConfig()
    assert not config.enable
    assert config.mace_model_path == "MACE-MP-0"
    assert config.uncertainty_threshold == 0.05
    assert config.sampling_count == 1000

def test_distillation_config_invalid() -> None:
    with pytest.raises(ValidationError):
        DistillationConfig(uncertainty_threshold=-0.1)
    with pytest.raises(ValidationError):
        DistillationConfig(sampling_count=0)

def test_active_learning_thresholds_defaults() -> None:
    config = ActiveLearningThresholds()
    assert config.threshold_call_dft == 0.05
    assert config.threshold_add_train == 0.02
    assert config.smooth_steps == 3

def test_active_learning_thresholds_invalid() -> None:
    with pytest.raises(ValidationError):
        ActiveLearningThresholds(threshold_call_dft=0)
    with pytest.raises(ValidationError):
        ActiveLearningThresholds(smooth_steps=0)

def test_cutout_config_defaults() -> None:
    config = CutoutConfig()
    assert config.core_radius == 3.0
    assert config.buffer_radius == 2.0
    assert config.pre_relax
    assert config.passivation

def test_cutout_config_invalid() -> None:
    with pytest.raises(ValidationError):
        CutoutConfig(core_radius=0)
    with pytest.raises(ValidationError):
        CutoutConfig(buffer_radius=-1.0)

def test_loop_strategy_config_defaults() -> None:
    config = LoopStrategyConfig()
    assert config.use_tiered_oracle
    assert config.incremental_update
    assert config.replay_buffer_size == 500
    assert config.surrogate_data_count == 1000

def test_loop_strategy_config_invalid() -> None:
    with pytest.raises(ValidationError):
        LoopStrategyConfig(replay_buffer_size=0)
    with pytest.raises(ValidationError):
        LoopStrategyConfig(surrogate_data_count=0)

def test_workflow_config_with_new_nested() -> None:
    config = WorkflowConfig(max_iterations=10)
    assert isinstance(config.distillation, DistillationConfig)
    assert isinstance(config.thresholds, ActiveLearningThresholds)
    assert isinstance(config.cutout, CutoutConfig)
    assert isinstance(config.loop_strategy, LoopStrategyConfig)
