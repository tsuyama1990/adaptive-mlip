from typing import Any
from unittest.mock import MagicMock

import numpy as np
from ase import Atoms


class MockDescriptorCalculator:
    """Mock for DescriptorCalculator to avoid heavy dscribe dependency in unit tests."""
    def __init__(self, config: Any) -> None:
        self.config = config

    def compute(self, atoms_list: list[Atoms]) -> np.ndarray:
        if not atoms_list:
            return np.array([])
        # Return random descriptors of fixed dimension (e.g. 10)
        n_structures = len(atoms_list)
        return np.random.rand(n_structures, 10)

class MockGenerator:
    """Mock for BaseGenerator."""
    def __init__(self, return_iter: Any) -> None:
        self.return_iter = return_iter

    def generate(self, n_candidates: int) -> Any:
        return self.return_iter
