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
    # Use epsilon=1.0, sigma=2.0 for predictable results
    oracle = MockOracle(epsilon=1.0, sigma=2.0)

    # Dimer at equilibrium distance (approx 2^(1/6) * sigma)
    # 2^(1/6) * 2.0 ~= 1.12246 * 2.0 = 2.2449
    r_eq = 2.2449
    # Use explicit positions for clear bond length
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, r_eq]])
    structure = AtomStructure(atoms=atoms)

    result_iter = oracle.compute(iter([structure]))
    result = next(result_iter)

    assert result.energy is not None
    assert result.forces is not None

    # Energy should be approx -epsilon (depth of well) at r_eq for a single pair?
    # Lennard-Jones potential V(r) = 4*epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
    # At r = r_eq = 2^(1/6)*sigma:
    # sigma/r = 2^(-1/6)
    # (sigma/r)^6 = 2^-1 = 0.5
    # (sigma/r)^12 = 0.25
    # V(r_eq) = 4 * 1.0 * [0.25 - 0.5] = 4 * (-0.25) = -1.0

    # Allow tolerance for ASE implementation details or precision
    assert np.isclose(result.energy, -1.0, atol=0.1)

    # Forces should be near zero at equilibrium
    forces = result.forces
    assert np.all(np.abs(forces) < 0.2)

    # Test repulsion (r < r_eq)
    r_short = 2.0
    atoms_short = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, r_short]])
    structure_short = AtomStructure(atoms=atoms_short)
    res_short = next(oracle.compute(iter([structure_short])))

    # Energy should be higher than min
    assert res_short.energy > -1.0

    # Forces should be repulsive (positive on 2nd atom along z?)
    # Force F = -dV/dr.
    # Repulsive force pushes atoms apart.
    # Atom 1 (z=0) pushed to -z, Atom 2 (z=r) pushed to +z.
    f_z_atom2 = res_short.forces[1][2]
    assert f_z_atom2 > 0.0
