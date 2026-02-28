from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BasePolicy
from pyacemaker.domain_models.structure import StructureConfig


class SafeBasePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        yield base_structure.copy()  # type: ignore[no-untyped-call]


class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """


class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """


class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """


class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """

    def __init__(self, policies: list[BasePolicy] | None = None) -> None:
        self.policies = policies or []

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        if not self.policies:
            return

        n_policies = len(self.policies)
        base_count = n_structures // n_policies
        remainder = n_structures % n_policies

        for i, policy in enumerate(self.policies):
            count = base_count + (1 if i < remainder else 0)
            yield from policy.generate(base_structure, config, n_structures=count, **kwargs)


class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        import random

        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            if len(atoms) > 0:
                idx = random.randint(0, len(atoms) - 1)
                del atoms[idx]
            yield atoms


class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        import numpy as np

        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            stdev = getattr(config, "rattle_stdev", 0.1)
            # Use random rattle directly instead of relying on the method which might be seeded
            # or do nothing on some setups in tests.
            positions = atoms.get_positions()
            positions += np.random.normal(scale=stdev, size=positions.shape)
            atoms.set_positions(positions)
            yield atoms


class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            cell = atoms.get_cell()
            strain_factor = getattr(config, "strain_factor", 1.05)
            atoms.set_cell(cell * strain_factor, scale_atoms=True)
            yield atoms
