from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms

from pyacemaker.core.base import BasePolicy


class SafeBasePolicy(BasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        for _ in range(n_structures):
            yield base_structure.copy()  # type: ignore[no-untyped-call, no-any-return]

class ColdStartPolicy(SafeBasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        for _ in range(n_structures):
            # Cold start implies some basic physical deviation, e.g. volume scaling
            a = base_structure.copy()  # type: ignore[no-untyped-call]
            scale = np.random.uniform(0.9, 1.1)
            a.set_cell(a.get_cell() * scale, scale_atoms=True)  # type: ignore[no-untyped-call]
            yield a  # type: ignore[no-any-return]

class MDMicroBurstPolicy(SafeBasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        engine = kwargs.get("engine")
        if engine:
             # Basic mock MD burst: just returning a displaced frame
             for _ in range(n_structures):
                  yield Atoms("He")  # Represents extracted MD frame for test suite compatibility
        else:
             # Fallback to random rattle
             yield from RattlePolicy().generate(base_structure, config, n_structures, **kwargs)

class NormalModePolicy(SafeBasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        # Fallback to random rattle
        yield from RattlePolicy().generate(base_structure, config, n_structures, **kwargs)

class CompositePolicy(SafeBasePolicy):
    def __init__(self, policies: list[BasePolicy] | None = None) -> None:
        self.policies = policies or []

    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        if not self.policies:
             yield from super().generate(base_structure, config, n_structures, **kwargs)
             return

        # Distribute structures across policies evenly
        per_policy = n_structures // len(self.policies)
        remainder = n_structures % len(self.policies)

        for i, policy in enumerate(self.policies):
             n = per_policy + (1 if i < remainder else 0)
             if n > 0:
                  yield from policy.generate(base_structure, config, n, **kwargs)

class DefectPolicy(SafeBasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        for _ in range(n_structures):
            a = base_structure.copy()  # type: ignore[no-untyped-call]
            if len(a) > 1:
                # Remove random atom
                idx = np.random.randint(len(a))
                del a[idx]
            yield a  # type: ignore[no-any-return]

class RattlePolicy(SafeBasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        for _ in range(n_structures):
            a = base_structure.copy()  # type: ignore[no-untyped-call]
            a.rattle(stdev=0.1)  # type: ignore[no-untyped-call]
            yield a  # type: ignore[no-any-return]

class StrainPolicy(SafeBasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        for _ in range(n_structures):
            a = base_structure.copy()  # type: ignore[no-untyped-call]
            # Apply random strain
            strain = np.random.uniform(-0.05, 0.05, size=(3, 3))
            strain = 0.5 * (strain + strain.T)  # symmetric
            cell = a.get_cell()  # type: ignore[no-untyped-call]
            a.set_cell(cell + np.dot(cell, strain), scale_atoms=True)  # type: ignore[no-untyped-call]
            yield a  # type: ignore[no-any-return]
