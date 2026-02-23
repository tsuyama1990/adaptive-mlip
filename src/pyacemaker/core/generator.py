from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig
from pyacemaker.utils.perturbations import rattle


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
        policy = PolicyFactory.get_policy(self.config)

        # Step 1: Base Structure Generation (Lazy)
        # We define composition here but don't call prediction yet
        composition = "".join(self.config.elements)

        # Validate policy configuration first
        if not isinstance(self.config.policy_name, ExplorationPolicy):
             msg = f"Invalid policy name: {self.config.policy_name}"
             raise TypeError(msg)

        # Step 2: Apply Policy (Streaming)
        # Create the supercell template lazily inside the generator.
        # This prevents materializing a potentially huge supercell if n_candidates is 0,
        # but more importantly, the base_supercell itself is just one object.
        # The true laziness comes from the policy yielding one by one.

        # Ensure we strictly follow the iterator protocol.
        def lazy_policy_stream() -> Iterator[Atoms]:
            # Lazy loading of base structure only when generator is started and first item requested
            try:
                base_structure = self.m3gnet.predict_structure(composition)
            except Exception as e:
                msg = f"Failed to generate base structure for {composition}: {e}"
                raise GeneratorError(msg) from e

            # Generate the base supercell template once.
            # We must materialize the base supercell to apply perturbations (rattle/strain).
            # While this object sits in memory, it is a single instance.
            # The streaming generator (yield) ensures we do not store n_candidates copies.
            # Thus, memory usage is O(Supercell_Size), not O(n_candidates * Supercell_Size).
            base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

            count = 0
            policy_iter = policy.generate(base_supercell, self.config, n_structures=n_candidates)

            # Verify it's an iterator to enforce streaming contract at runtime
            if not isinstance(policy_iter, Iterator):
                 # Convert iterable to iterator if needed
                 iter_policy = iter(policy_iter)
            else:
                 iter_policy = policy_iter

            for structure in iter_policy:
                if count >= n_candidates:
                    break
                if len(structure) == 0:
                    continue
                yield structure
                count += 1

        yield from lazy_policy_stream()

    def generate_local(self, base_structure: Atoms, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures by perturbing a base structure.
        Used in OTF loops to explore the local neighborhood of a high-uncertainty configuration.

        Args:
            base_structure: The reference structure to perturb.
            n_candidates: Number of structures to generate.

        Returns:
            Iterator yielding ASE Atoms objects.
        """
        if n_candidates <= 0:
            return

        # Use configured rattle_stdev, or default if not set (though Config enforces it)
        stdev = self.config.rattle_stdev

        # Always include the base structure itself as an anchor (spec says so)
        # But wait, BaseGenerator.generate_local contract?
        # The spec says "Select ... This sets S0 is always included (Anchor)".
        # ActiveSetSelector selects. Generator just generates candidates.
        # But for robustness, let's generate n_candidates.
        # If we want S0 in candidate pool, we can yield it first.
        # Spec says: "Generate Local Candidates ... (A) Normal Mode ... (C) Random Displacement".

        # We will yield perturbed structures.
        for _ in range(n_candidates):
            yield rattle(base_structure, stdev=stdev)
