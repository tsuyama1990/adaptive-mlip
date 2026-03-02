from pyacemaker.domain_models.workflow import ActiveLearningThresholds, CutoutConfig


def test_cycle03_thermal_noise_and_cutout():
    """
    Scenario 3: Phase 3 - Thermal Noise Rejection & Intelligent Cutout
    Verify the two-tier threshold schema and CutoutConfig.
    """
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=1.5,
        threshold_add_train=3.0,
        smooth_steps=5
    )
    assert thresholds.threshold_call_dft == 1.5
    assert thresholds.smooth_steps == 5

    cutout = CutoutConfig(
        core_radius=5.0,
        buffer_radius=10.0,
        enable_pre_relaxation=True,
        enable_passivation=True
    )
    assert cutout.core_radius == 5.0
    assert cutout.buffer_radius == 10.0

