from collections.abc import Iterator
from typing import Any

from ase import Atoms
from ase.calculators.lj import LennardJones

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models.constants import (
    MOCK_ORACLE_EPSILON,
    MOCK_ORACLE_RC,
    MOCK_ORACLE_SIGMA,
)


class MockOracle(BaseOracle):
    """
    A mock oracle for testing and UAT.
    Uses a simple Lennard-Jones potential to compute energies and forces quickly
    without requiring external DFT codes.

    Limitations:
    - Assumes all atoms interact with the same generic Lennard-Jones parameters.
    - Does not support species-dependent parameters in this simplified version.
    - Not suitable for accurate physics, only for pipeline verification.
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialize the MockOracle.

        Args:
            config: Optional configuration (ignored for mock).
        """
        # Default LJ parameters for "Generic" element
        self.sigma = MOCK_ORACLE_SIGMA
        self.epsilon = MOCK_ORACLE_EPSILON
        self.rc = MOCK_ORACLE_RC

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes properties using Lennard-Jones potential.
        """
        from logging import getLogger
        logger = getLogger(__name__)

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
            except (ValueError, ArithmeticError, RuntimeError) as e:
                # Catch known calculation errors and log them
                logger.error(f"MockOracle computation failed: {e}")
                # We do not yield the atom if calculation fails, effectively filtering it out.
                continue
            except Exception as e:
                # Propagate unexpected errors (programming bugs)
                logger.critical(f"Unexpected error in MockOracle: {e}")
                raise
