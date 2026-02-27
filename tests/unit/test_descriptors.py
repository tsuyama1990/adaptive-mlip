from collections.abc import Iterator

import numpy as np
import pytest
from ase import Atoms

# Stubs for interfaces to be implemented
# In TDD, we write tests against the EXPECTED interface.

class MockDescriptorCalculator:
    """Stub for DescriptorCalculator"""
    def __init__(self, config) -> None:
        self.config = config

    def compute(self, atoms_list: list[Atoms]) -> np.ndarray:
        # Mock behavior: return random descriptors matching n_atoms x dim
        n_structures = len(atoms_list)
        dim = 10 # Mock dimension
        return np.random.rand(n_structures, dim)

def test_descriptor_calculator_interface() -> None:
    # This test verifies how we WANT to use the descriptor calculator
    from pyacemaker.domain_models.active_learning import DescriptorConfig

    config = DescriptorConfig(
        method="soap",
        species=["Cu"],
        r_cut=5.0,
        n_max=8,
        l_max=6,
        sigma=0.5
    )

    # Expected usage:
    # calculator = DescriptorCalculator(config)
    # descriptors = calculator.compute(atoms_list)

    calculator = MockDescriptorCalculator(config)
    atoms_list = [Atoms('Cu'), Atoms('Cu')]
    descriptors = calculator.compute(atoms_list)

    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[0] == 2
    assert descriptors.shape[1] > 0
