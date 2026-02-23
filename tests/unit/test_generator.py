from itertools import islice

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StrainMode, StructureConfig


def test_cold_start_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.COLD_START,
        num_structures=10,  # Request 10
    )
    generator = StructureGenerator(config)
    # Use list() here is fine as we expect only 1
    structures = list(generator.generate(n_candidates=10))

    # Cold Start yields 1 structure regardless of n
    assert len(structures) == 1
    atoms = structures[0]
    assert isinstance(atoms, Atoms)
    assert len(atoms) > 0


def test_rattle_policy_streaming() -> None:
    """Verify that Rattle policy streams structures."""
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
    )
    generator = StructureGenerator(config)

    # We ask for 100, but only consume 2.
    # This verifies we don't crash or hang trying to compute 100 upfront.
    gen = generator.generate(n_candidates=100)

    first_two = list(islice(gen, 2))
    assert len(first_two) == 2
    assert isinstance(first_two[0], Atoms)
    assert isinstance(first_two[1], Atoms)

    # Ensure they are different
    assert not np.allclose(first_two[0].positions, first_two[1].positions)


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

    # Defect structures should have fewer atoms
    assert len(structures[0]) < len(base_atoms)


def test_strain_policy_volume() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.STRAIN,
        strain_mode=StrainMode.VOLUME,
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=2))
    assert len(structures) == 2

    # Volume strain changes volume, cell angles should remain same as base
    base_config = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    base_gen = StructureGenerator(base_config)
    base_atoms = next(base_gen.generate(1))

    cell = structures[0].get_cell()  # type: ignore[no-untyped-call]
    base_cell = base_atoms.get_cell()  # type: ignore[no-untyped-call]

    assert not np.isclose(cell.volume, base_cell.volume)
    assert np.allclose(cell.angles(), base_cell.angles())


def test_strain_policy_shear() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.STRAIN,
        strain_mode=StrainMode.SHEAR,
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=2))
    assert len(structures) == 2

    # Shear strain changes angles
    cell = structures[0].get_cell()  # type: ignore[no-untyped-call]

    # Generate base to compare
    base_config = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    base_gen = StructureGenerator(base_config)
    base_atoms = next(base_gen.generate(1))
    base_cell = base_atoms.get_cell()  # type: ignore[no-untyped-call]

    # Angles should change (randomly)
    # Extremely unlikely to be exactly the same
    assert not np.allclose(cell.angles(), base_cell.angles())


def test_strain_policy_mixed() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.STRAIN,
        strain_mode=StrainMode.MIXED,
    )
    generator = StructureGenerator(config)
    structures = list(generator.generate(n_candidates=2))
    assert len(structures) == 2

    # Mixed mode should likely change both volume and angles
    base_config = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    base_gen = StructureGenerator(base_config)
    base_atoms = next(base_gen.generate(1))

    cell = structures[0].get_cell()  # type: ignore[no-untyped-call]
    base_cell = base_atoms.get_cell()  # type: ignore[no-untyped-call]

    # Either volume or angles (or both) changed
    vol_diff = not np.isclose(cell.volume, base_cell.volume)
    angle_diff = not np.allclose(cell.angles(), base_cell.angles())

    assert vol_diff or angle_diff


def test_generator_invalid_composition() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.COLD_START,
    )
    generator = StructureGenerator(config)

    def mock_raise(comp: str) -> Atoms:
        msg = "Simulated failure"
        raise ValueError(msg)

    # Monkeypatch the wrapper method instance
    generator.m3gnet.predict_structure = mock_raise  # type: ignore

    with pytest.raises(RuntimeError, match="Failed to generate base structure"):
        next(generator.generate(1))
