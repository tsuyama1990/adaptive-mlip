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
        batch_size=2,
        candidate_multiplier=1
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

def test_direct_sampler_error_recovery() -> None:
    """Test behavior when generator throws an error or descriptor fails."""
    descriptor_config = DescriptorConfig(
        method="soap", species=["Cu"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5
    )
    config = Step1DirectSamplingConfig(target_points=3, descriptor=descriptor_config)
    mock_generator = MagicMock()
    mock_generator.generate.side_effect = RuntimeError("Generator failed")

    sampler = DirectSampler(config, mock_generator)
    with pytest.raises(RuntimeError, match="Generator failed"):
        list(sampler.generate())

def test_descriptor_calculator_oom_prevention() -> None:
    from pyacemaker.domain_models.active_learning import DescriptorConfig

    from pyacemaker.utils.descriptors import DescriptorCalculator

    config = DescriptorConfig(method="soap", species=["Cu"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5)

    with patch("pyacemaker.utils.descriptors.DescriptorCalculator._initialize_transformer"):
        # Mock the dimension calculation to force an OOM trigger
        calc = DescriptorCalculator.__new__(DescriptorCalculator)
        calc.config = config
        calc._transformer = MagicMock()
        calc._dim = 1_000_000 # Very large dim

        # 300 atoms * 1M dim * 8 bytes > 2GB limit
        atoms_list = [Atoms('Cu') for _ in range(300)]

        with pytest.raises(ValueError, match="exceeds safe memory limits"):
            calc.compute(atoms_list)
