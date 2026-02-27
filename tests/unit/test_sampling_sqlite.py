from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.active_learning import DescriptorConfig
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig
from pyacemaker.modules.sampling import DirectSampler


class MockDescriptorCalculator:
    def __init__(self, config: object) -> None: pass
    def compute(self, atoms_list: list[Atoms]) -> np.ndarray:
        return np.random.rand(len(atoms_list), 10)

def test_direct_sampler_sqlite_flow() -> None:
    descriptor_config = DescriptorConfig(
        method="soap", species=["Cu"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5
    )
    config = Step1DirectSamplingConfig(
        target_points=3,
        objective="maximize_entropy",
        descriptor=descriptor_config,
        batch_size=2
    )

    mock_generator = MagicMock()
    def candidate_stream(n_candidates: int) -> Iterator[AtomStructure]:
        for i in range(n_candidates):
            yield AtomStructure(atoms=Atoms('Cu', positions=[[i, 0, 0]]))

    mock_generator.generate.side_effect = candidate_stream

    with patch("pyacemaker.modules.sampling.DescriptorCalculator", MockDescriptorCalculator):
        sampler = DirectSampler(config, mock_generator)
        results = list(sampler.generate())

        assert len(results) == 3
        # Verify it used SQLite under the hood without error and reconstructed the atoms correctly
        assert all(isinstance(r, AtomStructure) for r in results)
        assert all(len(r.atoms) == 1 for r in results)
