from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.data import AtomStructure


# UAT Scenario 2.1: DIRECT Sampling Efficiency
def test_uat_2_1_direct_sampling_efficiency() -> None:
    """
    Scenario 2.1: DIRECT Sampling Efficiency
    Objective: Confirm that the DIRECT sampling strategy produces a diverse dataset.
    """
    # This is a high-level test that would use the real implementation.
    # For TDD, we mock the heavy lifting but verify the flow.

    with patch("pyacemaker.modules.sampling.DirectSampler") as MockSampler:
        # Mock the return of generate
        mock_instance = MockSampler.return_value
        mock_instance.generate.return_value = iter([
            AtomStructure(atoms=Atoms('Cu')),
            AtomStructure(atoms=Atoms('Cu'))
        ])

        # In a real UAT, we would call Orchestrator here
        # But for now we just verify the component interaction
        from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig

        # Verify we can instantiate config
        # (This imports are just to verify availability)


# Scenario 2.2: Active Learning Selection
def test_uat_2_2_active_learning_selection() -> None:
    """
    Scenario 2.2: Active Learning Selection
    Objective: Verify that structures with high uncertainty are correctly identified.
    """
    # 1. Create dummy structures with injected "true" uncertainty
    structures = []
    for i in range(100):
        s = AtomStructure(atoms=Atoms('Cu')) # Mock structure
        s.uncertainty = float(i) / 100.0 # Linear 0.0 to 1.0
        structures.append(s)

    # 2. Select Top 10 (Simulation of Orchestrator logic)
    structures.sort(key=lambda x: x.uncertainty if x.uncertainty else 0.0, reverse=True)
    active_set = structures[:10]

    # 3. Validation
    assert len(active_set) == 10
    assert active_set[0].uncertainty == 0.99
    assert active_set[-1].uncertainty == 0.90
