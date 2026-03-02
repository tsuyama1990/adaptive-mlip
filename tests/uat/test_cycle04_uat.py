from pyacemaker.domain_models.workflow import LoopStrategyConfig


def test_cycle04_hierarchical_fine_tuning():
    """
    Scenario 4: Phase 4 - Hierarchical Fine-Tuning & Seamless Resume
    Verify incremental update settings.
    """
    config = LoopStrategyConfig(
        use_tiered_oracle=True,
        incremental_update=True,
        replay_buffer_size=500,
        baseline_potential_type="LJ",
    )
    assert config.incremental_update is True
    assert config.replay_buffer_size == 500
