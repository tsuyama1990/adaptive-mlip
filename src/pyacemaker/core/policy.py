from collections.abc import Iterator
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BasePolicy


class SafeBasePolicy(BasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        Must be implemented by concrete classes to provide valid, physically perturbed structures.
        """
        raise NotImplementedError("Concrete policy must implement generation logic")

# Re-implement ColdStartPolicy and others that might have been overwritten or missing
class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        # Implementation of proper volume expansion/compression random generation goes here
        # Throw NotImplementedError to ensure developers complete it correctly rather than silently failing
        raise NotImplementedError("ColdStartPolicy generation not implemented")

class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        # Implementation for running short MD and sampling frames goes here
        raise NotImplementedError("MDMicroBurstPolicy generation not implemented")

class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        raise NotImplementedError("NormalModePolicy generation not implemented")


class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """
    def __init__(self, policies: list[BasePolicy] | None = None) -> None:
        self.policies = policies or []

    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        # Implementation to distribute generation across sub-policies goes here
        raise NotImplementedError("CompositePolicy generation not implemented")


class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        raise NotImplementedError("DefectPolicy generation not implemented")


class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        raise NotImplementedError("RattlePolicy generation not implemented")

class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        raise NotImplementedError("StrainPolicy generation not implemented")
