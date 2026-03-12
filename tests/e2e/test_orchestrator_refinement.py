from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseGenerator, BaseOracle


# Fake Components
class FakeGenerator(BaseGenerator):  # type: ignore[misc]
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
            yield getattr(base_structure, "copy")()


class FakeOracle(BaseOracle):  # type: ignore[misc]
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -5.0
            yield atoms
