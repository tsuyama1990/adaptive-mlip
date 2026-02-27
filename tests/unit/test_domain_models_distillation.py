import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.distillation import (
    DistillationConfig,
    Step1DirectSamplingConfig,
    Step2ActiveLearningConfig,
)


def test_step1_defaults() -> None:
    config = Step1DirectSamplingConfig()
    assert config.target_points == 100
    assert config.objective == "maximize_entropy"


def test_step1_validation() -> None:
    with pytest.raises(ValidationError):
        Step1DirectSamplingConfig(target_points=-1)


def test_step2_defaults() -> None:
    config = Step2ActiveLearningConfig()
    assert config.uncertainty_threshold == 0.8
    assert config.dft_calculator == "VASP"


def test_distillation_config_defaults() -> None:
    config = DistillationConfig()
    assert config.enable_mace_distillation is False
    assert isinstance(config.step1_direct_sampling, Step1DirectSamplingConfig)
    assert isinstance(config.step2_active_learning, Step2ActiveLearningConfig)


def test_distillation_config_override() -> None:
    config = DistillationConfig(
        enable_mace_distillation=True,
        step1_direct_sampling={"target_points": 500},
        step2_active_learning={"uncertainty_threshold": 0.5},
    )
    assert config.enable_mace_distillation is True
    assert config.step1_direct_sampling.target_points == 500
    assert config.step2_active_learning.uncertainty_threshold == 0.5
