from itertools import islice

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
    # Verify first few items using islice to avoid materializing all if N was large
    first_two = list(islice(stream, 2))

    assert len(first_two) == 2
    assert all(isinstance(s, Atoms) for s in first_two)

    # Verify chemistry
    s0 = first_two[0]
    symbols = s0.get_chemical_symbols()  # type: ignore[no-untyped-call]
    assert "Fe" in symbols
    assert "Pt" in symbols

    # Verify perturbation
    pos0 = first_two[0].positions
    pos1 = first_two[1].positions
    assert not np.allclose(pos0, pos1)

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
    pristine_atoms = next(pristine_gen.generate(1))

    assert len(defect_atoms) < len(pristine_atoms)
    assert np.allclose(defect_atoms.get_cell(), pristine_atoms.get_cell())  # type: ignore[no-untyped-call]
