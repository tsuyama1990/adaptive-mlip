
from pyacemaker.core.policy import (
    BasePolicy,
    ColdStartPolicy,
    DefectPolicy,
    RattlePolicy,
    StrainPolicy,
)
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


class PolicyFactory:
    """Factory for selecting and instantiating exploration policies."""

    @staticmethod
    def get_policy(config: StructureConfig) -> BasePolicy:
        """
        Selects the appropriate policy based on configuration.

        Args:
            config: Structure configuration.

        Returns:
            Instantiated policy object.

        Raises:
            ValueError: If the policy name is unknown.
        """
        policies: dict[ExplorationPolicy, type[BasePolicy]] = {
            ExplorationPolicy.COLD_START: ColdStartPolicy,
            ExplorationPolicy.RANDOM_RATTLE: RattlePolicy,
            ExplorationPolicy.STRAIN: StrainPolicy,
            ExplorationPolicy.DEFECTS: DefectPolicy,
        }

        policy_cls = policies.get(config.policy_name)
        if not policy_cls:
            msg = f"Unknown policy: {config.policy_name}"
            raise ValueError(msg)

        return policy_cls()
