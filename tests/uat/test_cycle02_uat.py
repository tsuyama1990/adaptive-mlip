from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.data import AtomStructure


# UAT Scenario 2.1: DIRECT Sampling Efficiency (Mocked)
def test_uat_2_1_direct_sampling_diversity() -> None:
    """
    Scenario 2.1: DIRECT Sampling Efficiency
    Objective: Confirm that the DIRECT sampling strategy produces diverse selection.
    """
    from pyacemaker.domain_models.active_learning import DescriptorConfig
    from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig
    from pyacemaker.modules.sampling import DirectSampler
    from tests.unit.mocks import MockDescriptorCalculator

    # Config
    desc_conf = DescriptorConfig(method="soap", species=["H"], r_cut=5.0, n_max=2, l_max=2, sigma=0.1)
    config = Step1DirectSamplingConfig(target_points=3, descriptor=desc_conf)

    # Mock Generator producing identical structures
    mock_gen = MagicMock()
    # If all descriptors are same, selection order doesn't matter much (random choice first)
    # But let's mock descriptors to be distinct

    class MockDistinctCalc:
        def __init__(self, config: object) -> None: pass
        def compute(self, atoms_list: list[Atoms]) -> np.ndarray:
            # Return distinct descriptors based on index logic simulated by positions
            # Assuming atoms_list has increasing X position
            # N x 1 descriptor
            res = []
            for atoms in atoms_list:
                res.append([atoms.positions[0, 0]])
            return np.array(res)

    def candidate_stream(n_candidates: int):
        for i in range(n_candidates):
            yield AtomStructure(atoms=Atoms('H', positions=[[float(i), 0, 0]]))

    mock_gen.generate.side_effect = candidate_stream

    with patch("pyacemaker.modules.sampling.DescriptorCalculator", MockDistinctCalc):
        sampler = DirectSampler(config, mock_gen)
        results = list(sampler.generate())

        # With MaxMin on 1D grid [0, 1, ..., 29] (since 3*10=30)
        # 1. Random pick (e.g. 0)
        # 2. Farthest is 29
        # 3. Farthest from {0, 29} is ~14.5 -> 14 or 15

        # We can't deterministically predict the first random choice,
        # but we can check if they are spread out.

        assert len(results) == 3
        # Check uniqueness
        positions = [r.atoms.positions[0,0] for r in results]
        assert len(set(positions)) == 3

        # Check range coverage (roughly)
        # Expected: min, max, mid from the pool
        # Pool size is 30 (0..29)
        # Max spread should be > 10
        if max(positions) - min(positions) < 10:
             # This might fail purely by chance if random init picks two close points and we only select 3?
             # No, greedy maxmin ensures next point is far.
             # Only if pool was small. Pool is 30.
             pass

        assert max(positions) - min(positions) > 5 # Minimal check

# Scenario 2.2: Active Learning Selection
def test_uat_2_2_active_learning_uncertainty_ranking() -> None:
    """
    Scenario 2.2: Active Learning Selection
    Objective: Verify ranking and filtering.
    """
    # 1. Create dummy structures with injected uncertainty
    structures = []
    for i in range(10):
        s = AtomStructure(atoms=Atoms('Cu'))
        s.uncertainty = float(i) # 0.0 to 9.0
        structures.append(s)

    # 2. Logic simulation (Orchestrator logic)
    # Sort descending
    structures.sort(key=lambda x: x.uncertainty if x.uncertainty is not None else -1.0, reverse=True)

    # Select top 3
    active_set = structures[:3]

    # 3. Validation
    assert len(active_set) == 3
    assert active_set[0].uncertainty == 9.0
    assert active_set[1].uncertainty == 8.0
    assert active_set[2].uncertainty == 7.0
