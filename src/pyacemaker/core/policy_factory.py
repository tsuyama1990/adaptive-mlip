from collections.abc import Callable
from typing import ClassVar

from pyacemaker.core.policy import (
    BasePolicy,
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


class PolicyRegistry:
    """Plugin registry for exploration policies ensuring Open/Closed compliance."""
    _policies: ClassVar[dict[ExplorationPolicy, type[BasePolicy]]] = {}
    _local_policies: ClassVar[dict[LocalGenerationStrategy, type[BasePolicy]]] = {}

    @classmethod
    def register(cls, policy_type: ExplorationPolicy) -> Callable[[type[BasePolicy]], type[BasePolicy]]:
        def wrapper(policy_cls: type[BasePolicy]) -> type[BasePolicy]:
            cls._policies[policy_type] = policy_cls
            return policy_cls
        return wrapper

    @classmethod
    def register_local(cls, strategy: LocalGenerationStrategy) -> Callable[[type[BasePolicy]], type[BasePolicy]]:
        def wrapper(policy_cls: type[BasePolicy]) -> type[BasePolicy]:
            cls._local_policies[strategy] = policy_cls
            return policy_cls
        return wrapper

    @classmethod
    def get(cls, policy_type: ExplorationPolicy) -> type[BasePolicy] | None:
        return cls._policies.get(policy_type)

    @classmethod
    def get_local(cls, strategy: LocalGenerationStrategy) -> type[BasePolicy] | None:
        return cls._local_policies.get(strategy)


PolicyRegistry.register(ExplorationPolicy.COLD_START)(ColdStartPolicy)
PolicyRegistry.register(ExplorationPolicy.RANDOM_RATTLE)(RattlePolicy)
PolicyRegistry.register(ExplorationPolicy.STRAIN)(StrainPolicy)
PolicyRegistry.register(ExplorationPolicy.DEFECTS)(DefectPolicy)

PolicyRegistry.register_local(LocalGenerationStrategy.RANDOM_DISPLACEMENT)(RattlePolicy)
PolicyRegistry.register_local(LocalGenerationStrategy.NORMAL_MODE)(NormalModePolicy)
PolicyRegistry.register_local(LocalGenerationStrategy.MD_MICRO_BURST)(MDMicroBurstPolicy)


class PolicyFactory:
    """Factory for selecting and instantiating exploration policies using the plugin registry."""

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
        active = config.active_policies
        if not active:
            active = [ExplorationPolicy.COLD_START]

        selected_policies = []
        for p_name in active:
            policy_cls = PolicyRegistry.get(p_name)
            if not policy_cls:
                msg = f"Unknown policy: {p_name}"
                raise ValueError(msg)
            selected_policies.append(policy_cls())

        if len(selected_policies) == 1:
            return selected_policies[0]

        return CompositePolicy(policies=selected_policies)

    @staticmethod
    def get_local_policy(strategy: LocalGenerationStrategy) -> BasePolicy:
        """
        Selects the appropriate policy for local generation based on strategy.

        Args:
            strategy: Local generation strategy enum.

        Returns:
            Instantiated policy object.
        """
        policy_cls = PolicyRegistry.get_local(strategy)
        if not policy_cls:
            msg = f"Unknown local strategy: {strategy}"
            raise ValueError(msg)

        return policy_cls()
