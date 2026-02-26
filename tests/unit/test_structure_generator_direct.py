from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.domain_models.structure import StructureConfig

# DirectSampler will be implemented in the next step
from pyacemaker.structure_generator.direct import DirectSampler


@pytest.fixture
def structure_config():
    return StructureConfig(
        elements=["Si"],
        supercell_size=[2, 2, 2],
        num_structures=5,
        r_cut=1.5
    )

def test_direct_sampler_initialization(structure_config):
    sampler = DirectSampler(config=structure_config)
    assert sampler.config == structure_config

def test_generate_returns_iterator(structure_config):
    sampler = DirectSampler(config=structure_config)
    structures = sampler.generate(n_candidates=5)

    # Check if it's an iterator
    assert hasattr(structures, "__next__")

    lst = list(structures)
    assert len(lst) == 5
    assert isinstance(lst[0], Atoms)
    # Check composition (Si8 for 2x2x2 supercell of Si diamond/fcc?)
    # Or just check symbol presence if random packing.
    # Assuming random packing fills box or places N atoms.
    # The spec says "Randomly places atoms in a box defined by supercell_size".
    assert "Si" in lst[0].get_chemical_symbols()

def test_generate_negative_count(structure_config):
    sampler = DirectSampler(config=structure_config)
    with pytest.raises(ValueError, match="n_candidates must be non-negative"):
        next(sampler.generate(n_candidates=-1))

def test_direct_sampler_update_config(structure_config):
    sampler = DirectSampler(config=structure_config)
    new_config = structure_config.model_copy()
    new_config.num_structures = 10
    sampler.update_config(new_config)
    assert sampler.config.num_structures == 10

def test_direct_sampler_update_config_invalid():
    sampler = DirectSampler(config=StructureConfig(elements=["Si"], supercell_size=[1,1,1]))
    with pytest.raises(TypeError):
        sampler.update_config({"invalid": "config"})

def test_generate_zero_count(structure_config):
    sampler = DirectSampler(config=structure_config)
    structures = list(sampler.generate(n_candidates=0))
    assert len(structures) == 0

def test_generate_fallback_bulk(structure_config):
    # Use an element likely not in ASE bulk (e.g. maybe "Og" or just mock bulk failure)
    # Mocking bulk is better.

    with patch("pyacemaker.structure_generator.direct.bulk", side_effect=Exception("No bulk")):
        sampler = DirectSampler(config=structure_config)
        structures = list(sampler.generate(n_candidates=1))
        assert len(structures) == 1
        # Should be box 10x10x10 * supercell [2,2,2] => 20x20x20
        # Actually supercell logic repeats cell vectors.
        # Fallback uses cell=[10,10,10].
        # 2x2x2 => 20x20x20.
        cell = structures[0].get_cell()
        assert cell[0,0] == 20.0

def test_generate_multiple_elements(structure_config):
    # Update config to multiple elements
    config = structure_config.model_copy()
    config.elements = ["Si", "C"]
    # Relax r_cut to ensure random packing succeeds quickly
    config.r_cut = 0.1
    sampler = DirectSampler(config=config)
    structures = list(sampler.generate(n_candidates=1))
    syms = structures[0].get_chemical_symbols()
    assert set(syms).issubset({"Si", "C"})

def test_generate_local(structure_config):
    sampler = DirectSampler(config=structure_config)
    # Just check it returns empty iterator
    res = list(sampler.generate_local(Atoms(), 10))
    assert res == []
