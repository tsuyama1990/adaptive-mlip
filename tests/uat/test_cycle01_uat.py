from pyacemaker.domain_models.workflow import DistillationConfig


def test_cycle01_distillation_schema():
    """
    Scenario 1: Phase 1 - Zero-Shot Distillation & Baseline Construction
    Verify that DistillationConfig is correctly instantiated and handles constraints.
    """
    config = DistillationConfig(
        enable=True, mace_model_path="MACE-MP-0", uncertainty_threshold=0.2, sampling_counts=500
    )
    assert config.enable is True
    assert config.uncertainty_threshold == 0.2
    assert config.sampling_counts == 500
