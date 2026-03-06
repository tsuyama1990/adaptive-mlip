from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BasePolicy
from pyacemaker.domain_models.structure import StructureConfig


class SafeBasePolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        yield base_structure

# Re-implement ColdStartPolicy and others that might have been overwritten or missing
class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)
        # Cold start logic stub

class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)

class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)

class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """
    def __init__(self, policies: list[BasePolicy] | None = None) -> None:
        self.policies = policies or []

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)

class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)

class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)

class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        yield from super().generate(base_structure, config, n_structures, **kwargs)
