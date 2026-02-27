from collections.abc import Iterator

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_cold_start_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        active_policies=[ExplorationPolicy.COLD_START],
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=10))

    # Cold Start yields 1 structure regardless of n
    assert len(structures) == 1
    structure = structures[0]
    assert isinstance(structure, AtomStructure)
    assert isinstance(structure.atoms, Atoms)


def test_rattle_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        active_policies=[ExplorationPolicy.RANDOM_RATTLE],
        rattle_stdev=0.1,
    )
    generator = StructureGenerator(config)

    # Check base structure first
    base_config = config.model_copy(update={"active_policies": [ExplorationPolicy.COLD_START]})
    base_gen = StructureGenerator(base_config)
    base_struct = next(base_gen.generate(1))
    base_atoms = base_struct.atoms

    structures = list(generator.generate(n_candidates=5))

    assert len(structures) == 5

    # Check if they are different objects
    assert structures[0] is not structures[1]

    # Verify positions are different between generated structures
    pos0 = structures[0].atoms.get_positions()
    pos1 = structures[1].atoms.get_positions()

    # Verify they are different from base
    assert not np.allclose(pos0, base_atoms.get_positions())

    # Verify they are different from each other
    assert not np.allclose(pos0, pos1)


def test_defect_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[3, 3, 3],
        active_policies=[ExplorationPolicy.DEFECTS],
        vacancy_rate=0.1,
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=5))

    assert len(structures) == 5

    base_config = config.model_copy(update={"active_policies": [ExplorationPolicy.COLD_START]})
    base_gen = StructureGenerator(base_config)
    base_struct = next(base_gen.generate(1))
    base_atoms = base_struct.atoms

    assert len(structures[0].atoms) < len(base_atoms)


def test_strain_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        active_policies=[ExplorationPolicy.STRAIN],
        strain_mode="volume",
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=5))

    assert len(structures) == 5

    base_config = config.model_copy(update={"active_policies": [ExplorationPolicy.COLD_START]})
    base_gen = StructureGenerator(base_config)
    base_struct = next(base_gen.generate(1))
    base_atoms = base_struct.atoms

    vol0 = structures[0].atoms.get_volume()  # type: ignore[no-untyped-call]
    base_vol = base_atoms.get_volume()  # type: ignore[no-untyped-call]
    assert vol0 != base_vol


def test_generator_invalid_composition() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.COLD_START],
    )
    generator = StructureGenerator(config)

    def mock_raise(comp: str) -> Atoms:
        msg = "Simulated failure"
        raise ValueError(msg)

    generator.m3gnet.predict_structure = mock_raise # type: ignore

    # Updated error message expectation
    with pytest.raises(GeneratorError, match="Base generator failed"):
        next(generator.generate(1))


def test_generate_local() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        active_policies=[ExplorationPolicy.COLD_START],
        rattle_stdev=0.1
    )
    generator = StructureGenerator(config)

    # Create dummy base structure
    base = Atoms("Fe2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)

    candidates = list(generator.generate_local(base, n_candidates=5))

    assert len(candidates) == 5
    for c in candidates:
        assert isinstance(c, AtomStructure)
        assert len(c.atoms) == 2
        # Check positions are slightly different
        assert not np.allclose(c.atoms.get_positions(), base.get_positions())
        # But cell is same
        assert np.allclose(c.atoms.get_cell(), base.get_cell())
