import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.distillation import (
    DistillationConfig,
    Step1DirectSamplingConfig,
    Step2ActiveLearningConfig,
)


def test_step1_defaults() -> None:
    config = Step1DirectSamplingConfig(descriptor={"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1})
    assert config.target_points == 100
    assert config.objective == "maximize_entropy"


def test_step1_validation() -> None:
    with pytest.raises(ValidationError):
        Step1DirectSamplingConfig(target_points=-1, descriptor={"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1})


def test_step2_defaults() -> None:
    config = Step2ActiveLearningConfig()
    assert config.uncertainty_threshold == 0.8
    assert config.dft_calculator == "VASP"


def test_distillation_config_defaults() -> None:
    config = DistillationConfig(enable_mace_distillation=False)
    assert config.enable_mace_distillation is False
    assert config.step1_direct_sampling is None
    assert config.step2_active_learning is None


def test_distillation_config_override() -> None:
    config = DistillationConfig(
        enable_mace_distillation=True,
        step1_direct_sampling={"target_points": 500, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
        step2_active_learning={"uncertainty_threshold": 0.5},
        step3_mace_finetune={"base_model": "test"},
    )
    assert config.enable_mace_distillation is True
    assert config.step1_direct_sampling.target_points == 500
    assert config.step2_active_learning.uncertainty_threshold == 0.5


def test_distillation_config_logic_validation() -> None:
    """Test custom logic validator."""
    with pytest.raises(ValidationError, match="Step 1 target points must be at least 10"):
        DistillationConfig(
            enable_mace_distillation=True,
            step1_direct_sampling={"target_points": 5, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
            step2_active_learning={"uncertainty_threshold": 0.5},
            step3_mace_finetune={"base_model": "test"},
        )


def test_distillation_config_logic_validation_disabled() -> None:
    """Validator should skip if disabled."""
    with pytest.raises(ValidationError, match="Distillation step configs must be None when enable_mace_distillation is False"):
        DistillationConfig(
            enable_mace_distillation=False,
            step1_direct_sampling={"target_points": 5, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
        )
