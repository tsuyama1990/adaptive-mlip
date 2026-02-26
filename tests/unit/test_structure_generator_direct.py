from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.domain_models.constants import DEFAULT_FALLBACK_CELL_SIZE
from pyacemaker.domain_models.structure import StructureConfig
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
    assert hasattr(structures, "__next__")
    lst = list(structures)
    assert len(lst) == 5
    assert isinstance(lst[0], Atoms)
    assert "Si" in lst[0].get_chemical_symbols()
    assert lst[0].info.get("provenance") == "DIRECT_SAMPLING"

def test_generate_negative_count(structure_config):
    sampler = DirectSampler(config=structure_config)
    with pytest.raises(ValueError, match="n_candidates must be non-negative"):
        next(sampler.generate(n_candidates=-1))

def test_generate_zero_count(structure_config):
    sampler = DirectSampler(config=structure_config)
    structures = list(sampler.generate(n_candidates=0))
    assert len(structures) == 0

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

def test_generate_fallback_bulk(structure_config):
    with patch("pyacemaker.structure_generator.direct.bulk", side_effect=Exception("No bulk")):
        sampler = DirectSampler(config=structure_config)
        structures = list(sampler.generate(n_candidates=1))
        assert len(structures) == 1
        # Fallback uses cell=[DEFAULT_FALLBACK_CELL_SIZE]*3.
        # 2x2x2 supercell => size * 2
        cell = structures[0].get_cell()
        expected_size = DEFAULT_FALLBACK_CELL_SIZE * 2
        assert cell[0,0] == expected_size

def test_generate_multiple_elements(structure_config):
    config = structure_config.model_copy()
    config.elements = ["Si", "C"]
    config.r_cut = 0.1
    sampler = DirectSampler(config=config)
    structures = list(sampler.generate(n_candidates=1))
    syms = structures[0].get_chemical_symbols()
    assert set(syms).issubset({"Si", "C"})

def test_generate_duplicate_elements_validation(structure_config):
    # Pydantic validates this on creation actually, but let's force it if possible
    # Or create a DirectSampler with a config that was manually mutated (bad practice but possible in python)
    # Actually StructureConfig validators run on init.
    # We can try to bypass pydantic validation by assignment?
    # No, Pydantic v2 validates assignment if configured.
    # But let's check if our DirectSampler catches it if config passed check but somehow has dupes (e.g. if validation was loose)
    # Our `DirectSampler.generate` has explicit check.

    # We need to construct a config with duplicates.
    # StructureConfig has validator `validate_elements` that checks duplicates.
    # So we can't create such config easily.
    # We can mock the config object.

    mock_config = MagicMock(spec=StructureConfig)
    mock_config.elements = ["Si", "Si"]

    sampler = DirectSampler(config=mock_config)
    with pytest.raises(ValueError, match="Duplicate elements"):
        next(sampler.generate(1))

def test_generate_invalid_rcut(structure_config):
    # Mock config to bypass pydantic ge=0.0 check if needed,
    # but r_cut has default 2.0 and gt=0.0.
    # DirectSampler checks r_cut <= 0.
    mock_config = MagicMock(spec=StructureConfig)
    mock_config.elements = ["Si"]
    mock_config.r_cut = -1.0

    sampler = DirectSampler(config=mock_config)
    with pytest.raises(ValueError, match="r_cut must be positive"):
        next(sampler.generate(1))

def test_generate_local(structure_config):
    sampler = DirectSampler(config=structure_config)
    res = list(sampler.generate_local(Atoms(), 10))
    assert res == []
