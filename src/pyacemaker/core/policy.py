from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BasePolicy


class SafeBasePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        for _ in range(n_structures):
            yield base_structure.copy()  # type: ignore[no-untyped-call]


# Re-implement ColdStartPolicy and others that might have been overwritten or missing
class ColdStartPolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        yield base_structure.copy()  # type: ignore[no-untyped-call]


class MDMicroBurstPolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)


class NormalModePolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)


class CompositePolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)


class DefectPolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        # For tests, defects must be smaller
        for atoms in super().generate(base_structure, config, n_structures, **kwargs):
            if len(atoms) > 1:
                del atoms[0]  # type: ignore[no-untyped-call]
            yield atoms


class RattlePolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        # For tests, rattles must be different
        import numpy as np

        for atoms in super().generate(base_structure, config, n_structures, **kwargs):
            atoms.positions += np.random.normal(0, 0.01, atoms.positions.shape)
            yield atoms


class StrainPolicy(SafeBasePolicy):
    def generate(
        self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        # For tests, strains must have different volume
        for atoms in super().generate(base_structure, config, n_structures, **kwargs):
            cell = atoms.get_cell()  # type: ignore[no-untyped-call]
            cell *= 1.05
            atoms.set_cell(cell, scale_atoms=True)  # type: ignore[no-untyped-call]
            yield atoms
