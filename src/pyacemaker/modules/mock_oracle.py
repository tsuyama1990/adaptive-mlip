from collections.abc import Iterator

import numpy as np
from ase.calculators.lj import LennardJones

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models.data import AtomStructure


class MockOracle(BaseOracle):
    """
    Mock Oracle implementation for testing and development.
    Uses a simple Lennard-Jones potential to compute energy and forces locally.
    Does not require external binaries or heavy dependencies.
    """

    def __init__(self, sigma: float = 2.5, epsilon: float = 1.0) -> None:
        """
        Initializes the Mock Oracle.

        Args:
            sigma: Lennard-Jones sigma parameter (default: 2.5 Angstrom).
            epsilon: Lennard-Jones epsilon parameter (default: 1.0 eV).
        """
        self.calc = LennardJones(sigma=sigma, epsilon=epsilon)

    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        """
        Computes properties for a stream of structures using Lennard-Jones.

        Args:
            structures: Iterator of AtomStructure objects.
            batch_size: Ignored for MockOracle (processing is fast and local).

        Yields:
            AtomStructure objects with energy, forces, and stress populated.
        """
        for structure in structures:
            atoms = structure.atoms.copy() # type: ignore[no-untyped-call]
            atoms.calc = self.calc

            try:
                energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
                forces = atoms.get_forces() # type: ignore[no-untyped-call]
                stress = atoms.get_stress() # type: ignore[no-untyped-call]
            except Exception:
                # Fallback for numerical instability or non-periodic stress
                energy = 0.0
                forces = np.zeros((len(atoms), 3))
                stress = np.zeros(6)

            # Update structure with computed properties
            structure.energy = float(energy)
            structure.forces = forces
            structure.stress = stress

            # Mock uncertainty for testing active learning loop
            structure.uncertainty = 0.0  # Zero uncertainty -> deterministic mock

            yield structure
