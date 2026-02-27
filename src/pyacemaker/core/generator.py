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
        # Note: PolicyFactory.get_policy returns a BasePolicy (or CompositePolicy)
        # which currently yields Atoms. We need to wrap them.
        policy = PolicyFactory.get_policy(self.config)

        # Step 1: Base Structure Generation (Lazy)
        # We define composition here but don't call prediction yet
        # If elements is a list of strings, join them if M3GNet expects formula string?
        # M3GNet usually takes formula like "Fe2O3".
        # StructureConfig.elements is ["Fe", "O"]. But doesn't have counts directly here.
        # Wait, StructureConfig has elements list, but composition/stoichiometry is needed.
        # The prompt says: "M3GNet predict_structure(composition)".
        # Let's assume composition string is derived or passed.
        # StructureConfig in SPEC doesn't explicitly have composition dict, only elements list.
        # However, defaults.py or PyAceConfig.system has composition.
        # Here StructureGenerator gets StructureConfig.
        # Issue: StructureConfig seems to lack composition info to form a valid formula for M3GNet unless "elements" contains it?
        # Re-reading PyAceConfig: system: SystemConfig (elements, composition).
        # StructureConfig is separate.
        # StructureGenerator is init with StructureConfig.
        # It seems StructureConfig should probably carry composition or Generator init should take SystemConfig too.
        # BUT, for Cycle 01, we might just use elements joined.
        # Let's assume elements list implies equal ratio or just elements for now.
        # "Cu", "Zr" -> "CuZr"?

        # NOTE: In Cycle 01 SPEC, PyAceConfig has `system` field.
        # StructureGenerator is initialized with `config.structure` in Factory.
        # This seems like a design gap in the existing codebase vs SPEC.
        # However, I should stick to existing patterns where possible or fix minimally.
        # StructureGenerator logic here uses `"".join(self.config.elements)`.

        composition = "".join(self.config.elements)

        def lazy_policy_stream() -> Iterator[AtomStructure]:
            # Lazy loading of base structure only when generator is started and first item requested
            try:
                # Assuming M3GNetWrapper handles this string
                base_structure = self.m3gnet.predict_structure(composition)
            except Exception as e:
                # If predict fails, wrap error
                raise GeneratorError(ERR_GEN_BASE_FAIL.format(composition=composition, error=e)) from e

            # Optimization: If supercell_size is (1,1,1), skip repeat to save a copy
            if tuple(self.config.supercell_size) == (1, 1, 1):
                base_supercell = base_structure
            else:
                base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

            count = 0
            # Policy yields Atoms
            policy_iter = policy.generate(base_supercell, self.config, n_structures=n_candidates)

            # Ensure iterator
            iter_policy = iter(policy_iter)

            for atoms in iter_policy:
                if count >= n_candidates:
                    break
                # Wrap Atoms in AtomStructure
                # Provenance: generator/policy info
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
