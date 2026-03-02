from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseGenerator, BaseOracle, BaseTrainer


# Fake Components
class FakeGenerator(BaseGenerator):
    def update_config(self, config: Any) -> None:
        pass

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        yield from []

    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        # Returns perturbations of base (S0)
        # We need to verify that base_structure passed here IS the extracted cluster.
        # We can tag it or check size.
        # Just yield copies for now.
        for _ in range(n_candidates):
            yield base_structure.copy()  # type: ignore[no-untyped-call]


class FakeOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -5.0
            yield atoms


class FakeTrainer(BaseTrainer):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:
        self.output_path.touch()
        return self.output_path


class FakeActiveSetSelector(ActiveSetSelector):
    def select(
        self, candidates: Any, potential_path: Any, n_select: int, anchor: Any = None
    ) -> Iterator[Atoms]:
        # Just return anchor and n_select-1 candidates
        if anchor:
            yield anchor
            n_select -= 1

        cands = list(candidates)
        for i in range(min(n_select, len(cands))):
            yield cands[i]


def test_orchestrator_refinement_logic():
    pass


def test_orchestrator_refinement_extraction_failure():
    pass
