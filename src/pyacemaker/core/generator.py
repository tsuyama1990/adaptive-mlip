from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.constants import ERR_GEN_BASE_FAIL, ERR_GEN_NCAND_NEG
from pyacemaker.domain_models.data import AtomStructure
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

    def generate(self, n_candidates: int) -> Iterator[AtomStructure]:
        """
        Generates candidate structures.

        This method returns an iterator to ensure streaming and O(1) memory usage.
        It uses the configured exploration policy to generate structures.

        Args:
            n_candidates: The number of candidate structures to generate.

        Yields:
            AtomStructure: Generated atomic structures.

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
        composition = "".join(self.config.elements)

        def lazy_policy_stream() -> Iterator[AtomStructure]:
            # Lazy loading of base structure only when generator is started and first item requested
            try:
                # Assuming M3GNetWrapper handles this string.
                # NOTE: M3GNet wrapper is used here. For large structures, predict_structure might take time,
                # but it returns a single ASE Atoms object. We only hold this single base structure in memory.
                # We DO NOT generate a list of n_candidates base structures.
                base_structure = self.m3gnet.predict_structure(composition)
            except Exception as e:
                raise GeneratorError(ERR_GEN_BASE_FAIL.format(composition=composition, error=e)) from e

            # Optimization: If supercell_size is (1,1,1), skip repeat to save a copy
            if tuple(self.config.supercell_size) == (1, 1, 1):
                base_supercell = base_structure
            else:
                base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

            # Streaming generation:
            # We call policy.generate which yields Atoms one by one.
            # We immediately wrap and yield AtomStructure.
            # No list is accumulated here.

            # Policy yields Atoms
            policy_iter = policy.generate(base_supercell, self.config, n_structures=n_candidates)

            # Ensure iterator
            iter_policy = iter(policy_iter)

            count = 0
            for atoms in iter_policy:
                if count >= n_candidates:
                    break

                # Wrap Atoms in AtomStructure
                provenance = {
                    "step": "generation",
                    "method": "policy_composite" # Simplified
                }

                yield AtomStructure(atoms=atoms, provenance=provenance)
                count += 1

        yield from lazy_policy_stream()

    def generate_local(self, base_structure: Atoms, n_candidates: int, **kwargs: Any) -> Iterator[AtomStructure]:
        """
        Generates candidate structures by perturbing a base structure.
        Used in OTF loops to explore the local neighborhood of a high-uncertainty configuration.
        Uses the configured local_generation_strategy.

        Args:
            base_structure: The reference structure to perturb.
            n_candidates: Number of structures to generate.
            **kwargs: Additional arguments (e.g., engine).

        Returns:
            Iterator yielding AtomStructure objects.
        """
        if n_candidates <= 0:
            return

        # Use PolicyFactory to get local policy
        strategy = self.config.local_generation_strategy
        policy = PolicyFactory.get_local_policy(strategy)

        # Generate using policy
        # Policy yields Atoms
        atoms_iter = policy.generate(base_structure, self.config, n_structures=n_candidates, **kwargs)

        for atoms in atoms_iter:
            provenance = {
                "step": "local_generation",
                "method": str(strategy)
            }
            yield AtomStructure(atoms=atoms, provenance=provenance)
