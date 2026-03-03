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
import typing
from pyacemaker.domain_models.structure import (
    ExplorationPolicy,
    LocalGenerationStrategy,
    StructureConfig,
)


class PolicyFactory:
    """Factory for selecting and instantiating exploration policies using a plugin registry."""

    _policies: dict[ExplorationPolicy, type[BasePolicy]] = {}
    _local_policies: dict[LocalGenerationStrategy, type[BasePolicy]] = {}

    @classmethod
    def register(cls, name: ExplorationPolicy) -> typing.Callable[[type[BasePolicy]], type[BasePolicy]]:
        def wrapper(policy_cls: type[BasePolicy]) -> type[BasePolicy]:
            cls._policies[name] = policy_cls
            return policy_cls
        return wrapper

    @classmethod
    def register_local(cls, name: LocalGenerationStrategy) -> typing.Callable[[type[BasePolicy]], type[BasePolicy]]:
        def wrapper(policy_cls: type[BasePolicy]) -> type[BasePolicy]:
            cls._local_policies[name] = policy_cls
            return policy_cls
        return wrapper

    @classmethod
    def get_policy(cls, config: StructureConfig) -> BasePolicy:
        """
        Selects the appropriate policy based on configuration (active_policies).
        Returns a CompositePolicy if multiple policies are active.
        """
        active = config.active_policies
        if not active:
            active = [ExplorationPolicy.COLD_START]

        selected_policies = []
        for p_name in active:
            policy_cls = cls._policies.get(p_name)
            if not policy_cls:
                msg = f"Unknown policy: {p_name}"
                raise ValueError(msg)
            selected_policies.append(policy_cls())

        if len(selected_policies) == 1:
            return selected_policies[0]

        return CompositePolicy()

    @classmethod
    def get_local_policy(cls, strategy: LocalGenerationStrategy) -> BasePolicy:
        """
        Selects the appropriate policy for local generation based on strategy.
        """
        policy_cls = cls._local_policies.get(strategy)
        if not policy_cls:
            msg = f"Unknown local strategy: {strategy}"
            raise ValueError(msg)

        return policy_cls()

# Register core policies
PolicyFactory.register(ExplorationPolicy.COLD_START)(ColdStartPolicy)
PolicyFactory.register(ExplorationPolicy.RANDOM_RATTLE)(RattlePolicy)
PolicyFactory.register(ExplorationPolicy.STRAIN)(StrainPolicy)
PolicyFactory.register(ExplorationPolicy.DEFECTS)(DefectPolicy)

PolicyFactory.register_local(LocalGenerationStrategy.RANDOM_DISPLACEMENT)(RattlePolicy)
PolicyFactory.register_local(LocalGenerationStrategy.NORMAL_MODE)(NormalModePolicy)
PolicyFactory.register_local(LocalGenerationStrategy.MD_MICRO_BURST)(MDMicroBurstPolicy)
