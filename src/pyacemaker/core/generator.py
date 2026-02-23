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
        policies: dict[ExplorationPolicy, type[BasePolicy]] = {
            ExplorationPolicy.COLD_START: ColdStartPolicy,
            ExplorationPolicy.RANDOM_RATTLE: RattlePolicy,
            ExplorationPolicy.STRAIN: StrainPolicy,
            ExplorationPolicy.DEFECTS: DefectPolicy,
        }

        policy_cls = policies.get(self.config.policy_name)
        if not policy_cls:
            msg = f"Unknown policy: {self.config.policy_name}"
            raise ValueError(msg)

        return policy_cls()

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
            ValueError: If n_candidates is negative or policy is invalid.
        """
        if n_candidates < 0:
            msg = f"n_candidates must be non-negative, got {n_candidates}"
            raise ValueError(msg)

        if n_candidates == 0:
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

        # Validate policy configuration first
        if not isinstance(self.config.policy_name, ExplorationPolicy):
             msg = f"Invalid policy name: {self.config.policy_name}"
             raise TypeError(msg)

        # Step 2: Apply Policy (Streaming)
        # Create the supercell template lazily.
        # We perform the repeat operation here, but it is only executed when the generator
        # is consumed (via next()).
        base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        # Stream directly from policy
        # Using 'yield from' or explicit loop ensures we strictly follow the iterator protocol.

        count = 0
        policy_iter = policy.generate(base_supercell, self.config, n_structures=n_candidates)

        # Verify it's an iterator to enforce streaming contract at runtime
        if not isinstance(policy_iter, Iterator):
             policy_iter = iter(policy_iter)

        for structure in policy_iter:
            if count >= n_candidates:
                break

            if len(structure) == 0:
                continue

            yield structure
            count += 1
