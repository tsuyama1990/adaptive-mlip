from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from pyacemaker.core.base import BaseOracle


class MockOracle(BaseOracle):
    """
    A mock oracle for testing and UAT.
    Uses a simple Lennard-Jones potential to compute energies and forces quickly
    without requiring external DFT codes.
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialize the MockOracle.

        Args:
            config: Optional configuration (ignored for mock).
        """
        # Default LJ parameters for "Generic" element
        self.sigma = 2.5
        self.epsilon = 1.0
        self.rc = 5.0

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes properties using Lennard-Jones potential.
        """
        for atoms in structures:
            # Create a fresh calculator for each structure to be safe
            # ASE LJ calculator is very lightweight
            calc = LennardJones(sigma=self.sigma, epsilon=self.epsilon, rc=self.rc)
            atoms.calc = calc

            # Trigger calculation
            try:
                e = atoms.get_potential_energy()
                f = atoms.get_forces()

                # Store in info/arrays as expected by pipeline
                atoms.info["energy"] = e
                atoms.arrays["forces"] = f
                atoms.info["provenance"] = "MOCK_ORACLE"

                # Clean up calculator to avoid pickling issues if any
                atoms.calc = None

                yield atoms
            except Exception:
                # In a real scenario, we might log this.
                # For mock, just skip or yield with None?
                # BaseOracle contract says yield with properties.
                # If calc fails, we skip.
                continue
