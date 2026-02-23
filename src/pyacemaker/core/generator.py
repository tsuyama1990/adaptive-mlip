from collections.abc import Iterator
from typing import Any

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

    def update_config(self, config: Any) -> None:
        """
        Updates the generator configuration.
        """
        if not isinstance(config, StructureConfig):
            msg = f"Expected StructureConfig, got {type(config)}"
            raise TypeError(msg)
        self.config = config

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
        # In Cold Start or perturbation, we need a base.

        composition = "".join(self.config.elements)

        try:
            base_structure = self.m3gnet.predict_structure(composition)
        except Exception as e:
            msg = f"Failed to generate base structure for {composition}: {e}"
            raise RuntimeError(msg) from e

        # Replicate base structure to supercell size
        # repeat() returns a new Atoms object. This is unavoidable for the base.
        # We do this once.
        base_structure = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        # Step 2: Apply Policy (Streaming)
        # Pass the base structure to the policy, which yields perturbed copies one by one.
        # This ensures we only have 1 base + 1 perturbed active in memory at a time.

        for i, structure in enumerate(policy.generate(base_structure, self.config, n_structures=n_candidates)):
            if i >= n_candidates:
                break

            # Data Integrity: Validate composition (basic check)
            if len(structure) == 0:
                 # Warn?
                 pass

            yield structure
