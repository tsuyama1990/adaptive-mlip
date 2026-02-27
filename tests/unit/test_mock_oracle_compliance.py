from collections.abc import Iterator

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.modules.mock_oracle import MockOracle


def test_mock_oracle_compliance() -> None:
    """Verify MockOracle implements BaseOracle interface."""
    assert issubclass(MockOracle, BaseOracle)
    oracle = MockOracle()
    assert isinstance(oracle, BaseOracle)


def test_mock_oracle_type_validation() -> None:
    """Verify MockOracle strictly validates input types."""
    oracle = MockOracle()

    # Pass list instead of Iterator
    with pytest.raises(TypeError, match="Expected Iterator"):
        next(oracle.compute([AtomStructure(atoms=Atoms("H"))])) # type: ignore

    # Pass Iterator of wrong type
    def wrong_iter() -> Iterator[str]:
        yield "not a structure"

    with pytest.raises(TypeError, match="Expected AtomStructure"):
        next(oracle.compute(wrong_iter())) # type: ignore


def test_mock_oracle_compute_logic() -> None:
    """Verify MockOracle computes reasonable values using LJ."""
    oracle = MockOracle(epsilon=1.0, sigma=2.0)

    # Dimer at equilibrium distance (approx 2^(1/6) * sigma)
    # 2^(1/6) * 2.0 ~= 1.122 * 2.0 = 2.244
    r = 2.244
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, r]])
    # MockOracle logic copies atoms and sets calc.
    # We need to make sure calc runs without error.
    # ASE LJ requires 'sigma' and 'epsilon' which we passed.
    # It might also check atomic numbers or symbols? LJ usually generic.

    structure = AtomStructure(atoms=atoms)

    result_iter = oracle.compute(iter([structure]))
    result = next(result_iter)

    assert result.energy is not None
    assert result.forces is not None
    assert result.stress is not None

    # Energy should be approx -epsilon (depth of well) at min for single pair
    # Total energy of pair = 4*eps * (-0.25) = -eps.
    # Wait, LJ minimum is -epsilon at r_min.
    # So Energy ~ -1.0.

    # Check if close to -1.0
    assert np.isclose(result.energy, -1.0, atol=0.1)

    # Forces should be low at equilibrium
    forces = result.forces
    assert np.all(np.abs(forces) < 1.0)
