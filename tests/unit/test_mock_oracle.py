from collections.abc import Iterator

import numpy as np
from ase import Atoms

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models.data import AtomStructure


class MockOracleImplementation(BaseOracle):
    """
    Temporary mock implementation for testing interface compliance
    before the real module is implemented.
    """
    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        for s in structures:
            # Simple dummy calculation
            s.energy = -1.0 * len(s.atoms)
            s.forces = np.zeros((len(s.atoms), 3))
            yield s


def test_mock_oracle_interface_compliance() -> None:
    """Verify that MockOracle implementation adheres to the new BaseOracle interface."""
    oracle = MockOracleImplementation()

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    input_structure = AtomStructure(atoms=atoms)

    # Must accept iterator of AtomStructure
    input_iter = iter([input_structure])

    # Must return iterator of AtomStructure
    output_iter = oracle.compute(input_iter)

    result = next(output_iter)

    assert isinstance(result, AtomStructure)
    assert result.energy is not None
    assert result.forces is not None
    assert result.energy == -2.0
