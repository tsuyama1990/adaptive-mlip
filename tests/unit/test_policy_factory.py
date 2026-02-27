
from pyacemaker.core.policy import (
    ColdStartPolicy,
    CompositePolicy,
    DefectPolicy,
    RattlePolicy,
    StrainPolicy,
)
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_get_policy_single_rattle() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.RANDOM_RATTLE]
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, RattlePolicy)


def test_get_policy_single_strain() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.STRAIN]
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, StrainPolicy)


def test_get_policy_single_defect() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.DEFECTS]
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, DefectPolicy)


def test_get_policy_single_cold_start() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.COLD_START]
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, ColdStartPolicy)


def test_get_policy_composite() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[
            ExplorationPolicy.RANDOM_RATTLE,
            ExplorationPolicy.STRAIN
        ]
    )
    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, CompositePolicy)
    assert len(policy.policies) == 2
    assert isinstance(policy.policies[0], RattlePolicy)
    assert isinstance(policy.policies[1], StrainPolicy)


def test_get_policy_legacy_compatibility() -> None:
    """Test that legacy 'policy_name' field (deprecated) is correctly synced."""
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        policy_name=ExplorationPolicy.DEFECTS
        # active_policies defaults to [COLD_START], but sync_policy_fields validator
        # should override it if policy_name is explicit?
        # Wait, sync_policy_fields updates active_policies based on policy_name if set?
        # Let's check domain_models/structure.py logic.
        # Logic: if policy_name is set and not in active_policies, overwrite active_policies = [policy_name]
    )

    # We re-validate to ensure validator runs if we constructed manually
    # Pydantic validates on init

    assert config.active_policies == [ExplorationPolicy.DEFECTS]

    policy = PolicyFactory.get_policy(config)
    assert isinstance(policy, DefectPolicy)
