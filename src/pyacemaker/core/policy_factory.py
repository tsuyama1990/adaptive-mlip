
from pyacemaker.core.base import BasePolicy
from pyacemaker.core.policy import (
    ColdStartPolicy,
    CompositePolicy,
    DefectPolicy,
    MDMicroBurstPolicy,
    NormalModePolicy,
    RandomDisplacementPolicy,
    RattlePolicy,
    StrainPolicy,
)
from pyacemaker.domain_models.structure import (
    ExplorationPolicy,
    LocalGenerationStrategy,
    StructureConfig,
)


class PolicyFactory:
    """
    Factory to create exploration policies.
    """

    @staticmethod
    def get_policy(config: StructureConfig) -> BasePolicy:
        """
        Creates a composite policy based on active_policies list in config.
        """
        policies: list[BasePolicy] = []

        for policy_name in config.active_policies:
            if policy_name == ExplorationPolicy.COLD_START:
                policies.append(ColdStartPolicy())
            elif policy_name == ExplorationPolicy.RANDOM_RATTLE:
                policies.append(RattlePolicy())
            elif policy_name == ExplorationPolicy.STRAIN:
                policies.append(StrainPolicy())
            elif policy_name == ExplorationPolicy.DEFECTS:
                policies.append(DefectPolicy())
            else:
                # Should be caught by Pydantic validation
                pass

        if not policies:
            # Default to ColdStart if empty? Pydantic ensures non-empty default list or validation?
            # Config default is [COLD_START]
            return ColdStartPolicy()

        if len(policies) == 1:
            return policies[0]

        return CompositePolicy(policies)

    @staticmethod
    def get_local_policy(strategy: LocalGenerationStrategy) -> BasePolicy:
        """
        Creates a policy for local generation strategy.
        """
        if strategy == LocalGenerationStrategy.RANDOM_DISPLACEMENT:
            return RandomDisplacementPolicy()
        if strategy == LocalGenerationStrategy.NORMAL_MODE:
            return NormalModePolicy()
        if strategy == LocalGenerationStrategy.MD_MICRO_BURST:
            return MDMicroBurstPolicy()

        return RandomDisplacementPolicy()
