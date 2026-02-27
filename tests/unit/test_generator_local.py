import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_generate_local_rattle() -> None:
    """Test local generation with Random Rattle (default)."""
    base = Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        local_generation_strategy="random_displacement",
        rattle_stdev=0.1,
    )
    generator = StructureGenerator(config)

    # Should yield n_candidates
    results = list(generator.generate_local(base, n_candidates=5))
    assert len(results) == 5

    for res in results:
        # Check structure similarity (cell same, positions moved)
        assert np.allclose(res.atoms.cell, base.cell)
        assert not np.allclose(res.atoms.positions, base.positions)
        assert res.provenance["method"] == "random_displacement"


def test_generate_local_md_burst() -> None:
    """Test local generation with MD Micro Burst (Mock/Fallback)."""
    base = Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        local_generation_strategy="md_micro_burst",
    )
    generator = StructureGenerator(config)

    # Even if MD not fully implemented, it should return AtomStructures
    results = list(generator.generate_local(base, n_candidates=2))
    assert len(results) == 2
    assert results[0].provenance["method"] == "md_micro_burst"


def test_generate_local_normal_mode() -> None:
    """Test local generation with Normal Mode (Mock/Fallback)."""
    base = Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        local_generation_strategy="normal_mode",
    )
    generator = StructureGenerator(config)

    results = list(generator.generate_local(base, n_candidates=2))
    assert len(results) == 2
    assert results[0].provenance["method"] == "normal_mode"
