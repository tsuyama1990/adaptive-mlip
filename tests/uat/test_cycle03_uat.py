import numpy as np
from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_uat_03_01_generate_candidates() -> None:
    """
    Scenario 03-01: Generate Candidates
    Objective: Verify that the system can generate a set of perturbed structures from a base composition.
    """
    # 1. Preparation
    config = StructureConfig(
        elements=["Fe", "Pt"],  # Composition FePt
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
        num_structures=10,
    )

    generator = StructureGenerator(config)

    # 2. Action
    # Use streaming consumption instead of list()
    stream = generator.generate(n_candidates=10)

    # 3. Expectation
    # Use pure streaming, do not materialize lists
    s0 = next(stream)
    s1 = next(stream)

    assert isinstance(s0, Atoms)
    assert isinstance(s1, Atoms)

    # Verify chemistry
    symbols = s0.get_chemical_symbols()  # type: ignore[no-untyped-call]
    assert "Fe" in symbols
    assert "Pt" in symbols

    # Verify perturbation (compare random samples to avoid full array materialization if large)
    # Just checking first atom's position is enough to verify they aren't identical
    assert not np.allclose(s0.positions[0], s1.positions[0])

    # Verify we can consume the rest without keeping them
    remaining_count = sum(1 for _ in stream)
    assert remaining_count == 8  # 10 - 2


def test_uat_03_02_defect_generation() -> None:
    """
    Scenario 03-02: Defect Generation
    Objective: Verify that the system can introduce vacancies.
    """
    # 1. Preparation
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[4, 4, 4],
        policy_name=ExplorationPolicy.DEFECTS,
        vacancy_rate=0.05,
    )
    generator = StructureGenerator(config)

    # 2. Action
    # Stream one
    stream = generator.generate(n_candidates=1)
    defect_atoms = next(stream)

    # 3. Expectation
    # Get pristine count
    pristine_config = config.model_copy(
        update={
            "policy_name": ExplorationPolicy.COLD_START,
            "active_policies": [ExplorationPolicy.COLD_START],
            "vacancy_rate": 0.0,
        }
    )
    pristine_gen = StructureGenerator(pristine_config)
    pristine_stream = pristine_gen.generate(1)
    pristine_atoms = next(pristine_stream)

    # Compare scalar properties to avoid array materialization
    assert len(defect_atoms) < len(pristine_atoms)

    # Compare scalar volume instead of full cell array
    vol_defect = defect_atoms.get_volume()  # type: ignore[no-untyped-call]
    vol_pristine = pristine_atoms.get_volume()  # type: ignore[no-untyped-call]
    assert abs(vol_defect - vol_pristine) < 1e-6
