from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.constants import ERR_GEN_BASE_FAIL, ERR_GEN_NCAND_NEG
from pyacemaker.domain_models.structure import StructureConfig


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
            raise ValueError(ERR_GEN_NCAND_NEG.format(n=n_candidates))

        if n_candidates == 0:
            return

        # Policy Selection
        # Uses active_policies via PolicyFactory
        policy = PolicyFactory.get_policy(self.config)

        # Step 1: Base Structure Generation (Lazy)
        # We define composition here but don't call prediction yet
        composition = "".join(self.config.elements)

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
                raise GeneratorError(ERR_GEN_BASE_FAIL.format(composition=composition, error=e)) from e

            # Generate the base supercell template once.
            # We must materialize the base supercell to apply perturbations (rattle/strain).
            # While this object sits in memory, it is a single instance.
            # The streaming generator (yield) ensures we do not store n_candidates copies.
            # Thus, memory usage is O(Supercell_Size), not O(n_candidates * Supercell_Size).
            # We ensure this materialization happens strictly inside the generator (lazy).

            # Optimization: If supercell_size is (1,1,1), skip repeat to save a copy
            if tuple(self.config.supercell_size) == (1, 1, 1):
                base_supercell = base_structure
            else:
                base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

            count = 0
            policy_iter = policy.generate(
                base_structure=base_supercell, config=self.config, n_structures=n_candidates
            )

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

    def generate_local(self, base_structure: Atoms, n_candidates: int, **kwargs: Any) -> Iterator[Atoms]:
        """
        Generates candidate structures by perturbing a base structure.
        Used in OTF loops to explore the local neighborhood of a high-uncertainty configuration.
        Uses the configured local_generation_strategy.

        Args:
            base_structure: The reference structure to perturb.
            n_candidates: Number of structures to generate.
            **kwargs: Additional arguments (e.g., engine).

        Returns:
            Iterator yielding ASE Atoms objects.
        """
        if n_candidates <= 0:
            return

        # Use PolicyFactory to get local policy
        strategy = self.config.local_generation_strategy
        policy = PolicyFactory.get_local_policy(strategy)

        # Generate using policy
        # Pass kwargs (e.g. engine) to allow advanced policies like MD Micro Burst
        yield from policy.generate(
            base_structure=base_structure,
            config=self.config,
            n_structures=n_candidates,
            **kwargs,
        )
