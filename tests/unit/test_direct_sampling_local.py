import pytest
from ase import Atoms

from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig
from pyacemaker.structure_generator.direct import DirectSampler


@pytest.fixture
def structure_config():
    return StructureConfig(
        elements=["Cu"],
        supercell_size=[2, 2, 2],
        num_structures=2,
        r_cut=1.5,
        active_policies=[ExplorationPolicy.COLD_START]
    )

def test_generate_local_returns_empty(structure_config):
    """
    Test that generate_local returns an empty iterator for now (not implemented).
    This confirms the intended stub behavior for Cycle 01.
    """
    sampler = DirectSampler(structure_config)
    dummy_atoms = Atoms("Cu")

    # Verify it returns an iterator
    gen = sampler.generate_local(dummy_atoms, n_candidates=5)

    # Verify it's empty safely
    count = 0
    for _ in gen:
        count += 1
    assert count == 0
