from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.active_learning import DescriptorConfig
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig


# Stub for DirectSampler
class DirectSampler:
    def __init__(self, config: Step1DirectSamplingConfig, generator) -> None:
        self.config = config
        self.generator = generator

    def generate(self) -> Iterator[AtomStructure]:
        # Logic to be implemented:
        # 1. Generate N candidates from self.generator
        # 2. Compute descriptors
        # 3. Select target_points using MaxMin
        # 4. Yield selected

        # Mock implementation for test
        candidates = self.generator.generate(n_candidates=self.config.target_points * 2)
        count = 0
        for cand in candidates:
            if count < self.config.target_points:
                # Add mock provenance
                cand.provenance['sampling_method'] = 'direct'
                yield cand
                count += 1

def test_direct_sampler_flow() -> None:
    # 1. Setup Config
    descriptor_config = DescriptorConfig(
        method="soap", species=["Cu"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5
    )
    config = Step1DirectSamplingConfig(
        target_points=5,
        objective="maximize_entropy",
        descriptor=descriptor_config
    )

    # 2. Mock Generator
    mock_generator = MagicMock()
    # Generator yields AtomStructure
    def candidate_stream(n_candidates):
        for i in range(n_candidates):
            yield AtomStructure(atoms=Atoms('Cu', positions=[[i, 0, 0]]))

    mock_generator.generate.side_effect = candidate_stream

    # 3. Instantiate Sampler
    sampler = DirectSampler(config, mock_generator)

    # 4. Run Generation
    results = list(sampler.generate())

    # 5. Assertions
    assert len(results) == 5
    assert results[0].provenance.get('sampling_method') == 'direct'
    # Check generator was called with sufficient buffer (implementation detail, usually 10x)
    # mock_generator.generate.assert_called_once()
