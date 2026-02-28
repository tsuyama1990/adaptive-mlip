from pyacemaker.core.base import BasePolicy
from pyacemaker.core.policy import (
    ColdStartPolicy,
    CompositePolicy,
    DefectPolicy,
    MDMicroBurstPolicy,
    NormalModePolicy,
    RattlePolicy,
    StrainPolicy,
)
from pyacemaker.domain_models.structure import (
    ExplorationPolicy,
    LocalGenerationStrategy,
    StructureConfig,
)


class PolicyFactory:
    """Factory for selecting and instantiating exploration policies."""

    @staticmethod
    def get_policy(config: StructureConfig) -> BasePolicy:
        """
        Selects the appropriate policy based on configuration (active_policies).
        Returns a CompositePolicy if multiple policies are active.

        Args:
            config: Structure configuration.

        Returns:
            Instantiated policy object.

        Raises:
            ValueError: If any policy name is unknown.
        """
        policies_map: dict[ExplorationPolicy, type[BasePolicy]] = {
            ExplorationPolicy.COLD_START: ColdStartPolicy,
            ExplorationPolicy.RANDOM_RATTLE: RattlePolicy,
            ExplorationPolicy.STRAIN: StrainPolicy,
            ExplorationPolicy.DEFECTS: DefectPolicy,
        }

        active = config.active_policies
        if not active:
            # Fallback to default if empty (should be handled by Pydantic default though)
            active = [ExplorationPolicy.COLD_START]

        selected_policies = []
        for p_name in active:
            policy_cls = policies_map.get(p_name)
            if not policy_cls:
                msg = f"Unknown policy: {p_name}"
                raise ValueError(msg)
            selected_policies.append(policy_cls())

        if len(selected_policies) == 1:
            return selected_policies[0]

        return CompositePolicy(selected_policies)

    @staticmethod
    def get_local_policy(strategy: LocalGenerationStrategy) -> BasePolicy:
        """
        Selects the appropriate policy for local generation based on strategy.

        Args:
            strategy: Local generation strategy enum.

        Returns:
            Instantiated policy object.
        """
        local_map: dict[LocalGenerationStrategy, type[BasePolicy]] = {
            LocalGenerationStrategy.RANDOM_DISPLACEMENT: RattlePolicy,
            LocalGenerationStrategy.NORMAL_MODE: NormalModePolicy,
            LocalGenerationStrategy.MD_MICRO_BURST: MDMicroBurstPolicy,
        }

        policy_cls = local_map.get(strategy)
        if not policy_cls:
            msg = f"Unknown local strategy: {strategy}"
            raise ValueError(msg)

        return policy_cls()
