from collections.abc import Iterator

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
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
    Generates candidate structures using the configured exploration policy.
    Uses M3GNet (via ColdStartPolicy) to establish a base structure,
    then applies perturbations (Rattle, Strain, Defects).
    """

    def __init__(self, config: StructureConfig) -> None:
        self.config = config
        self.cold_start_policy = ColdStartPolicy(config)
        self.policy = self._get_policy(config.policy_name)

    def _get_policy(self, name: ExplorationPolicy) -> BasePolicy:
        if name == ExplorationPolicy.COLD_START:
            return self.cold_start_policy
        if name == ExplorationPolicy.RANDOM_RATTLE:
            return RattlePolicy(self.config)
        if name == ExplorationPolicy.STRAIN:
            return StrainPolicy(self.config)
        if name == ExplorationPolicy.DEFECTS:
            return DefectPolicy(self.config)
        msg = f"Unknown policy: {name}"
        raise ValueError(msg)

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates 'n_candidates' structures.

        1. Obtains a base structure using ColdStartPolicy (M3GNet/Mock).
        2. Applies supercell expansion.
        3. Applies the active policy (perturbation) to the base structure.
        """
        # Note: ColdStartPolicy.apply() ignores input and returns new structure from composition
        base = self.cold_start_policy.apply()

        # Step 2: Apply Supercell
        if self.config.supercell_size:
            base *= self.config.supercell_size

        # Step 3: Yield candidates
        for _ in range(n_candidates):
            if self.config.policy_name == ExplorationPolicy.COLD_START:
                # To address OOM Risk: We yield a freshly generated structure to avoid memory
                # retention of modified objects, rather than copying a potentially large
                # 'base' object repeatedly or risking shared mutable state.
                fresh_base = self.cold_start_policy.apply()
                if self.config.supercell_size:
                    fresh_base *= self.config.supercell_size
                yield fresh_base
            else:
                # Apply perturbation policies (Rattle, Strain, Defects)
                # These policies copy the input structure internally, so 'base' remains clean.
                yield self.policy.apply(base)
