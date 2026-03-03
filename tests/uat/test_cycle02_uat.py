from pyacemaker.domain_models.workflow import LoopStrategyConfig


def test_cycle02_loop_strategy() -> None:
    """
    Scenario 2: Phase 2 - Physical Validation & Auto-Retraining
    Verify that LoopStrategyConfig manages replay_buffer_size and uses TieredOracle.
    """
    config = LoopStrategyConfig(
        use_tiered_oracle=True,
        incremental_update=False,
        replay_buffer_size=2000,
        baseline_potential_type="LJ",
    )
    assert config.use_tiered_oracle is True
    assert config.replay_buffer_size == 2000
