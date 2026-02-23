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

        This allows adaptive policies to modify generation parameters at runtime.

        Args:
            config: New configuration object (must be an instance of StructureConfig).

        Raises:
            TypeError: If the provided config is not a StructureConfig instance.
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
        It uses the configured exploration policy to generate structures.

        Args:
            n_candidates: The number of candidate structures to generate.

        Yields:
            Atoms: Generated atomic structures.

        Raises:
            RuntimeError: If base structure generation fails.
            ValueError: If the configured policy is invalid.
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

        # Validate policy configuration first
        if not isinstance(self.config.policy_name, ExplorationPolicy):
             msg = f"Invalid policy name: {self.config.policy_name}"
             raise TypeError(msg)

        # Step 2: Apply Policy (Streaming)
        # We generate the supercell once to use as a template.
        # This is created lazily when the generator is consumed (i.e. on first next() call).
        base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        # Stream directly from policy
        # The policy uses the base_supercell to generate candidates.
        # We ensure we yield immediately to prevent memory accumulation.
        # Using 'yield from' or explicit loop ensures we strictly follow the iterator protocol.

        count = 0
        policy_iter = policy.generate(base_supercell, self.config, n_structures=n_candidates)

        # Verify it's an iterator to enforce streaming contract at runtime
        if not isinstance(policy_iter, Iterator):
             # Just in case a policy implementation returns a list
             policy_iter = iter(policy_iter)

        for structure in policy_iter:
            if count >= n_candidates:
                break

            if len(structure) == 0:
                continue

            yield structure
            count += 1
