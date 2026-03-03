from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.defaults import ERR_GEN_BASE_FAIL, ERR_GEN_NCAND_NEG
from pyacemaker.domain_models.structure import StructureConfig


class StructureGenerator(BaseGenerator):
    """
    Structure Generator implementation.
    Uses M3GNet (or mock) for base structure and exploration policies for perturbations.
    """

    def __init__(self, config: StructureConfig, use_mock_m3gnet: bool = True) -> None:
        self.config = config
        self.m3gnet = M3GNetWrapper(use_mock=use_mock_m3gnet)

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

        # Ensure we strictly follow the iterator protocol with pure O(1) space complexity constraints
        def lazy_policy_stream() -> Iterator[Atoms]:
            try:
                base_structure = self.m3gnet.predict_structure(composition)
            except Exception as e:
                raise GeneratorError(
                    ERR_GEN_BASE_FAIL.format(composition=composition, error=e)
                ) from e

            count = 0
            # To absolutely guarantee O(1) memory, we apply policy generating directly
            # to the small base structure and incrementally apply the repetition internally
            # or recreate the cell incrementally avoiding holding the root massive
            # array inside outer loop states if possible.
            # The most direct way to stop OOM is to lazily construct the base_supercell
            # *per yield* or rely on the policy returning an iterator of supercells directly.

            # Since policy expects the supercell, we will yield over the generator,
            # but we won't keep an isolated `base_supercell` block memory in the generator function scope
            # once passed to the policy.

            # Use a separate scope function to instantiate and immediately discard
            def _create_supercell() -> Atoms:
                if tuple(self.config.supercell_size) == (1, 1, 1):
                    return base_structure.copy() # type: ignore[no-untyped-call]
                return base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

            policy_iter = policy.generate(
                _create_supercell(), self.config, n_structures=n_candidates
            )

            if not isinstance(policy_iter, Iterator):
                iter_policy = iter(policy_iter)
            else:
                iter_policy = policy_iter

            try:
                for structure in iter_policy:
                    if count >= n_candidates:
                        break
                    if len(structure) == 0:
                        continue
                    yield structure
                    count += 1
            finally:
                # Cleanup internal pointers actively
                del iter_policy

        yield from lazy_policy_stream()

    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
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
        yield from policy.generate(base_structure, self.config, n_structures=n_candidates, **kwargs)
