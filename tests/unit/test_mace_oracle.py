from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models.data import AtomStructure


# Stub for MaceOracle
class MaceOracle(BaseOracle):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device

    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        for s in structures:
            # Mock computation
            s.energy = -10.0
            s.forces = np.zeros((len(s.atoms), 3))
            s.uncertainty = 0.5 # Mock uncertainty
            s.stress = np.zeros(6) # Mock stress
            yield s

def test_mace_oracle_compute() -> None:
    # 1. Setup
    oracle = MaceOracle(model_path="fake_model.pt")

    # 2. Input Data
    structures = [
        AtomStructure(atoms=Atoms('Cu', positions=[[0,0,0]])),
        AtomStructure(atoms=Atoms('Cu', positions=[[1,1,1]]))
    ]

    # 3. Compute
    results = list(oracle.compute(iter(structures)))

    # 4. Assertions
    assert len(results) == 2
    for res in results:
        assert res.energy is not None
        assert res.forces is not None
        assert res.uncertainty is not None
        assert res.uncertainty == 0.5
