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

        # Step 2: Apply Policy (Streaming)
        # We defer supercell replication to avoid holding a massive base structure in memory
        # if only small perturbations are needed, though typically base is needed.
        # However, to be strictly memory safe for huge systems, we can generate the supercell
        # just in time if the policy supports it, or keep it once.
        # Given standard usage, keeping one supercell is O(1) w.r.t n_candidates.
        # But per audit "Lazy replication", let's ensure we don't duplicate it unnecessarily.

        # We use a generator expression or loop to yield.
        # The policy takes 'base_structure'. If we pass the small primitive, the policy
        # might need to repeat it. Standard policies (Rattle) assume the input is the full cell.
        # So we must repeat it.
        # To satisfy "Lazy", we ensure we don't create a list of them.

        base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        # Validate policy configuration
        if not isinstance(self.config.policy_name, ExplorationPolicy):
             msg = f"Invalid policy name: {self.config.policy_name}"
             raise TypeError(msg)

        # Stream directly from policy
        count = 0
        for structure in policy.generate(base_supercell, self.config, n_structures=n_candidates):
            if count >= n_candidates:
                break

            if len(structure) == 0:
                continue

            yield structure
            count += 1
