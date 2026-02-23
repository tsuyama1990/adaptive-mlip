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
        # Should be caught by Pydantic validation, but for safety:
        msg = f"Unknown policy: {self.config.policy_name}"
        raise ValueError(msg)

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures.
        """
        if n_candidates <= 0:
            return

        # Step 1: Get base structure
        # Composition is derived from elements list (simple join for now)
        # Stoichiometry is implicitly 1:1:1... based on unique elements list
        composition = "".join(self.config.elements)

        try:
            base_structure = self.m3gnet.predict_structure(composition)
        except Exception as e:
            msg = f"Failed to generate base structure for {composition}: {e}"
            raise RuntimeError(msg) from e

        # Replicate base structure to supercell size
        # We assume M3GNet returns a unit cell (primitive or conventional)
        # repeat() returns a new Atoms object
        base_structure = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        # Step 2: Apply Policy
        policy = self._get_policy()

        # Yield from policy generator
        # Note: ColdStartPolicy ignores n_candidates and yields 1 structure (as duplicates are useless)
        # Other policies yield n_candidates variations.
        yield from policy.generate(base_structure, self.config, n_structures=n_candidates)
