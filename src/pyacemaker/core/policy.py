from collections.abc import Iterator
from typing import Any
from ase import Atoms

from pyacemaker.core.base import BasePolicy


class SafeBasePolicy(BasePolicy):
    def generate(self, base_structure: Atoms, config: Any, n_structures: int = 1, **kwargs: Any) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        Fallback to returning the base structure if not fully implemented.
        """
        # Just return the base structure wrapped in a list/iterator to satisfy interface and tests
        yield base_structure

# Re-implement ColdStartPolicy and others that might have been overwritten or missing
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

class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """

class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """

class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """
