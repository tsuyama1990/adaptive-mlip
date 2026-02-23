from collections.abc import Iterator

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.core.policy import (
    BasePolicy,
    ColdStartPolicy,
    DefectPolicy,
    RattlePolicy,
    StrainPolicy,
)
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


class StructureGenerator(BaseGenerator):
    """
    Structure Generator implementation.
    Uses M3GNet (or mock) for base structure and exploration policies for perturbations.
    """

    def __init__(self, config: StructureConfig) -> None:
        self.config = config
        self.m3gnet = M3GNetWrapper()

    def _get_policy(self) -> BasePolicy:
        """Selects the appropriate policy based on configuration."""
        if self.config.policy_name == ExplorationPolicy.COLD_START:
            return ColdStartPolicy()
        if self.config.policy_name == ExplorationPolicy.RANDOM_RATTLE:
            return RattlePolicy()
        if self.config.policy_name == ExplorationPolicy.STRAIN:
            return StrainPolicy()
        if self.config.policy_name == ExplorationPolicy.DEFECTS:
            return DefectPolicy()
        msg = f"Unknown policy: {self.config.policy_name}"
        raise ValueError(msg)

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures.
        This method returns an iterator to ensure streaming and O(1) memory usage.
        """
        if n_candidates <= 0:
            return

        # Policy Selection
        policy = self._get_policy()

        # Step 1: Base Structure Generation (Streaming)
        composition = "".join(self.config.elements)

        try:
            base_structure = self.m3gnet.predict_structure(composition)
        except Exception as e:
            msg = f"Failed to generate base structure for {composition}: {e}"
            raise RuntimeError(msg) from e

        # We do NOT repeat the structure here. We pass the unit cell (or primitive cell)
        # to the policy. The policy is responsible for repeating it efficiently.
        # This avoids holding a potentially large supercell in memory unnecessarily.

        # Step 2: Apply Policy (Streaming)
        for i, structure in enumerate(
            policy.generate(base_structure, self.config, n_structures=n_candidates)
        ):
            if i >= n_candidates:
                break

            # Data Integrity: Validate composition (basic check)
            if len(structure) == 0:
                # Log warning?
                continue

            yield structure
