from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseGenerator
from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.core.policy_factory import PolicyFactory
from pyacemaker.domain_models.constants import ERR_GEN_BASE_FAIL, ERR_GEN_NCAND_NEG
from pyacemaker.domain_models.structure import StructureConfig


class BaseStructureGenerator:
    """Generates the foundational atomic structure."""
    def __init__(self, config: StructureConfig) -> None:
        self.config = config
        self.m3gnet = M3GNetWrapper()

    def get_base_supercell(self) -> Atoms:
        composition = "".join(self.config.elements)
        if len(composition) > 100:
            msg = f"Composition string is excessively long ({len(composition)} chars), which may cause issues for M3GNet."
            raise ValueError(msg)

        try:
            base_structure = self.m3gnet.predict_structure(composition)
        except Exception as e:
            raise GeneratorError(ERR_GEN_BASE_FAIL.format(composition=composition, error=e)) from e

        if tuple(self.config.supercell_size) == (1, 1, 1):
            return base_structure

        return base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]


class PolicyOrchestrator:
    """Orchestrates structural policies to apply to the base structure."""
    def __init__(self, config: StructureConfig) -> None:
        self.config = config

    def apply(self, base_supercell: Atoms, n_candidates: int) -> Iterator[Atoms]:
        policy = PolicyFactory.get_policy(self.config)
        policy_iter = policy.generate(base_supercell, self.config, n_structures=n_candidates)

        iter_policy = iter(policy_iter) if not isinstance(policy_iter, Iterator) else policy_iter

        count = 0
        for structure in iter_policy:
            if count >= n_candidates:
                break
            if len(structure) == 0:
                continue
            yield structure
            count += 1

    def apply_local(self, base_structure: Atoms, n_candidates: int, engine: Any | None = None, potential: str | Path | None = None) -> Iterator[Atoms]:
        strategy = self.config.local_generation_strategy
        policy = PolicyFactory.get_local_policy(strategy)
        yield from policy.generate(base_structure, self.config, n_structures=n_candidates, engine=engine, potential=potential)


class StructureGenerator(BaseGenerator):
    """
    Structure Generator implementation.
    Uses M3GNet (or mock) for base structure and exploration policies for perturbations.
    """

    def __init__(self, config: StructureConfig) -> None:
        self.config = config
        self.base_generator = BaseStructureGenerator(config)
        self.orchestrator = PolicyOrchestrator(config)

        # Keep m3gnet reference for backwards compatibility with tests
        self.m3gnet = self.base_generator.m3gnet

    def update_config(self, config: Any) -> None:
        if not isinstance(config, StructureConfig):
            msg = f"Expected StructureConfig, got {type(config)}"
            raise TypeError(msg)
        self.config = config
        self.base_generator.config = config
        self.orchestrator.config = config

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        if n_candidates < 0:
            raise ValueError(ERR_GEN_NCAND_NEG.format(n=n_candidates))
        if n_candidates == 0:
            return

        def lazy_policy_stream() -> Iterator[Atoms]:
            base_supercell = self.base_generator.get_base_supercell()
            yield from self.orchestrator.apply(base_supercell, n_candidates)

        yield from lazy_policy_stream()

    def generate_local(self, base_structure: Atoms, n_candidates: int, engine: Any | None = None, potential: str | Path | None = None) -> Iterator[Atoms]:
        if n_candidates <= 0:
            return
        yield from self.orchestrator.apply_local(base_structure, n_candidates, engine, potential)
