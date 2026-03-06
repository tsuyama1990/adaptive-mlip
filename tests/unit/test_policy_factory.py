from pyacemaker.core.policy import CompositePolicy, MDMicroBurstPolicy, RattlePolicy, StrainPolicy
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.structure import (
    ExplorationPolicy,
    LocalGenerationStrategy,
    StructureConfig,
)


def test_get_policy_single():
    config = StructureConfig(
        elements=["H"], supercell_size=[1, 1, 1], active_policies=[ExplorationPolicy.RANDOM_RATTLE]
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, RattlePolicy)
    assert not isinstance(policy, CompositePolicy)


def test_get_policy_composite():
    config = StructureConfig(
        elements=["H"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.RANDOM_RATTLE, ExplorationPolicy.STRAIN],
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, CompositePolicy)
    assert len(policy.policies) == 2
    assert isinstance(policy.policies[0], RattlePolicy)
    assert isinstance(policy.policies[1], StrainPolicy)


def test_get_local_policy():
    policy = PolicyFactory.get_local_policy(LocalGenerationStrategy.MD_MICRO_BURST)
    assert isinstance(policy, MDMicroBurstPolicy)

    policy = PolicyFactory.get_local_policy(LocalGenerationStrategy.RANDOM_DISPLACEMENT)
    assert isinstance(policy, RattlePolicy)
