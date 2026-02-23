import numpy as np
from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_generator_yields_n_structures() -> None:
    config = StructureConfig(
        elements=["Fe"], supercell_size=[1, 1, 1], policy_name=ExplorationPolicy.COLD_START
    )
    generator = StructureGenerator(config)
    n = 5
    results = list(generator.generate(n))

    assert len(results) == n
    for atoms in results:
        assert isinstance(atoms, Atoms)
        # Should be Fe or Fe1 depending on implementation
        assert "Fe" in atoms.get_chemical_formula()


def test_generator_uses_rattle_policy_for_diversity() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
    )
    generator = StructureGenerator(config)
    results = list(generator.generate(5))

    assert len(results) == 5

    # Check diversity
    pos0 = results[0].get_positions()
    pos1 = results[1].get_positions()

    # Very unlikely to be identical with random rattle
    assert not np.allclose(pos0, pos1)
