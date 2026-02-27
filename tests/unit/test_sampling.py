from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.active_learning import DescriptorConfig
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig
from pyacemaker.modules.sampling import DirectSampler
from tests.unit.mocks import MockDescriptorCalculator


def test_direct_sampler_streaming() -> None:
    # 1. Setup Config
    descriptor_config = DescriptorConfig(
        method="soap", species=["Cu"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5
    )
    config = Step1DirectSamplingConfig(
        target_points=5,
        objective="maximize_entropy",
        descriptor=descriptor_config,
        batch_size=2 # Small batch size to test batching logic
    )

    # 2. Mock Generator
    mock_generator = MagicMock()
    # Generator yields AtomStructure
    def candidate_stream(n_candidates: int) -> Iterator[AtomStructure]:
        for i in range(n_candidates):
            # Create dummy atoms
            yield AtomStructure(atoms=Atoms('Cu', positions=[[i, 0, 0]]))

    mock_generator.generate.side_effect = candidate_stream

    # 3. Instantiate Sampler with Mock Descriptor Calculator
    with patch("pyacemaker.modules.sampling.DescriptorCalculator", SideEffect=MockDescriptorCalculator) as MockCalc:
        MockCalc.return_value = MockDescriptorCalculator(descriptor_config)

        sampler = DirectSampler(config, mock_generator)

        # 4. Run Generation
        # We expect 50 candidates (10x target) to be generated and processed
        results = list(sampler.generate())

        # 5. Assertions
        assert len(results) == 5
        assert results[0].provenance.get('sampling_method') == 'direct_maxmin'

        # Verify generator called with multiplier
        mock_generator.generate.assert_called_with(n_candidates=50)

def test_direct_sampler_no_candidates() -> None:
    descriptor_config = DescriptorConfig(
        method="soap", species=["Cu"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5
    )
    config = Step1DirectSamplingConfig(target_points=5, descriptor=descriptor_config)
    mock_generator = MagicMock()
    mock_generator.generate.return_value = iter([]) # Empty stream

    with patch("pyacemaker.modules.sampling.DescriptorCalculator", SideEffect=MockDescriptorCalculator):
        sampler = DirectSampler(config, mock_generator)
        results = list(sampler.generate())
        assert len(results) == 0
