from itertools import islice

import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig
from pyacemaker.structure_generator.direct import DirectSampler


@pytest.fixture
def structure_config():
    # Use smaller supercell or fewer atoms to make random packing easier for tests
    return StructureConfig(
        elements=["Cu"],
        supercell_size=[2, 2, 2], # 4 atoms (fcc primitive * 2*2*2 = 4*8 = 32 atoms?) No, primitive is 1 atom.
        # bulk("Cu") returns primitive cell (1 atom) or conventional (4 atoms)?
        # ase.build.bulk("Cu") returns primitive fcc (1 atom).
        # repeat([2,2,2]) -> 8 atoms.
        # 8 atoms in 2x2x2 primitive volume is dense.
        # But we expand volume by 1.5x in generator now.
        # Let's use a generous r_cut for testing but feasible.
        num_structures=5,
        r_cut=1.5, # Smaller r_cut to allow packing
        active_policies=[ExplorationPolicy.COLD_START]
    )

def test_direct_sampler_initialization(structure_config):
    """Test DirectSampler initialization."""
    sampler = DirectSampler(structure_config)
    assert sampler.config == structure_config

def test_generate_returns_iterator(structure_config):
    """Test generate method returns an iterator of Atoms."""
    sampler = DirectSampler(structure_config)
    # Use islice to avoid consuming potential infinite streams (though direct is finite)
    # and to verify it works as an iterator
    gen = sampler.generate(n_candidates=3)
    structures = list(islice(gen, 3))
    assert len(structures) == 3, f"Expected 3 structures, got {len(structures)}"
    assert all(isinstance(s, Atoms) for s in structures)

def test_generated_structures_provenance(structure_config):
    """Test generated structures have correct metadata."""
    sampler = DirectSampler(structure_config)
    # Ensure generator produces something
    gen = sampler.generate(n_candidates=1)
    try:
        structure = next(gen)
    except StopIteration:
        pytest.fail("Generator yielded no structures.")

    assert structure.info["provenance"] == "DIRECT_SAMPLING"
    assert structure.info["method"] == "random_packing"

def test_no_overlaps(structure_config):
    """Test generated structures respect r_cut (hard-sphere constraint)."""
    # Increase r_cut to make overlap more likely if logic is broken
    structure_config.r_cut = 1.8
    sampler = DirectSampler(structure_config)

    # Use explicit loop to avoid list() OOM risk on large datasets (though here N=3 is small)
    count = 0
    for structure in sampler.generate(n_candidates=3):
        count += 1
        # neighbor_list("d", ...) returns distances < r_cut
        # Since we enforce d >= r_cut, this list should be empty (or only self-interactions if any)
        # neighbor_list typically excludes self-interaction unless configured otherwise.

        # Check strict inequality used in generator
        distances = neighbor_list("d", structure, structure_config.r_cut - 0.001)
        assert len(distances) == 0, f"Found atoms closer than {structure_config.r_cut}"

    assert count == 3, "Failed to generate structures for overlap test"

def test_multi_element_generation():
    """Test generation with multiple elements."""
    config = StructureConfig(
        elements=["Fe", "Pt"],
        supercell_size=[2, 2, 2],
        num_structures=2,
        r_cut=1.5
    )
    sampler = DirectSampler(config)
    gen = sampler.generate(n_candidates=1)
    try:
        structure = next(gen)
    except StopIteration:
        pytest.fail("Generator yielded no structures for multi-element test.")

    symbols = set(structure.get_chemical_symbols())
    assert "Fe" in symbols or "Pt" in symbols

def test_generate_respects_n_candidates(structure_config):
    """Test generator yields exactly n_candidates."""
    sampler = DirectSampler(structure_config)
    count = 0
    for _ in sampler.generate(n_candidates=2):
        count += 1
    assert count == 2

def test_generate_zero_candidates(structure_config):
    """Test generate with 0 candidates."""
    sampler = DirectSampler(structure_config)
    # Using list() on 0 items is safe and correct way to verify emptiness
    structures = list(sampler.generate(n_candidates=0))
    assert len(structures) == 0

def test_invalid_n_candidates(structure_config):
    """Test generate with negative candidates raises ValueError."""
    sampler = DirectSampler(structure_config)
    with pytest.raises(ValueError, match="n_candidates must be non-negative"):
        next(sampler.generate(n_candidates=-1))

def test_update_config(structure_config):
    """Test update_config updates the configuration."""
    sampler = DirectSampler(structure_config)
    new_config = structure_config.model_copy()
    new_config.num_structures = 100
    sampler.update_config(new_config)
    assert sampler.config.num_structures == 100

def test_update_config_invalid():
    """Test update_config raises TypeError for invalid config."""
    config = StructureConfig(elements=["Cu"], supercell_size=[2,2,2])
    sampler = DirectSampler(config)
    with pytest.raises(TypeError):
        sampler.update_config({"invalid": "dict"}) # type: ignore

def test_generate_impossible_packing(structure_config):
    """Test that generate raises ValueError for impossible packing density."""
    # Create a scenario where atoms cannot fit
    # High r_cut, small cell, many atoms
    structure_config.r_cut = 5.0
    structure_config.supercell_size = [1, 1, 1]
    # Cu bulk has a ~ 3.6 A.
    # With 1.5 scaling -> 5.4 A.
    # Volume ~ 157.
    # Atom volume (r=2.5) ~ 65.
    # 4 atoms (fcc primitive * 1) -> 4 * 65 = 260 > 157.
    # Packing fraction > 1.0. Should fail.

    sampler = DirectSampler(structure_config)
    with pytest.raises(ValueError, match="Impossible packing density"):
        next(sampler.generate(n_candidates=1))
