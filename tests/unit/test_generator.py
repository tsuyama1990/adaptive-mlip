import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_cold_start_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.COLD_START,
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=10))

    # Cold Start yields 1 structure regardless of n
    assert len(structures) == 1
    atoms = structures[0]
    assert isinstance(atoms, Atoms)


def test_rattle_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
    )
    generator = StructureGenerator(config)

    # Check base structure first
    base_gen = StructureGenerator(config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START}))
    base = next(base_gen.generate(1))

    structures = list(generator.generate(n_candidates=5))

    assert len(structures) == 5

    # Check if they are different objects
    assert structures[0] is not structures[1]

    # Verify positions are different between generated structures
    pos0 = structures[0].positions.copy()
    pos1 = structures[1].positions.copy()

    # Verify they are different from base
    assert not np.allclose(pos0, base.positions)

    # Verify they are different from each other
    assert not np.allclose(pos0, pos1)


def test_defect_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[3, 3, 3],
        policy_name=ExplorationPolicy.DEFECTS,
        vacancy_rate=0.1,
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=5))

    assert len(structures) == 5

    base_config = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    base_gen = StructureGenerator(base_config)
    base_atoms = next(base_gen.generate(1))

    assert len(structures[0]) < len(base_atoms)


def test_strain_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.STRAIN,
        strain_mode="volume",
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=5))

    assert len(structures) == 5

    base_config = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    base_gen = StructureGenerator(base_config)
    base_atoms = next(base_gen.generate(1))

    vol0 = structures[0].get_volume()  # type: ignore[no-untyped-call]
    base_vol = base_atoms.get_volume()  # type: ignore[no-untyped-call]
    assert vol0 != base_vol


def test_generator_invalid_composition() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        policy_name=ExplorationPolicy.COLD_START,
    )
    generator = StructureGenerator(config)

    def mock_raise(comp: str) -> Atoms:
        msg = "Simulated failure"
        raise ValueError(msg)

    generator.m3gnet.predict_structure = mock_raise # type: ignore

    with pytest.raises(RuntimeError, match="Failed to generate base structure"):
        next(generator.generate(1))
