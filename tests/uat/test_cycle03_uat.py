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
    # Set config.yaml equivalent (using Pydantic model directly here)
    config = StructureConfig(
        elements=["Fe", "Pt"],  # Composition FePt
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
        num_structures=10,
    )

    generator = StructureGenerator(config)

    # 2. Action
    structures = list(generator.generate(n_candidates=10))

    # 3. Expectation
    # The system returns a list of 10 ase.Atoms objects.
    assert len(structures) == 10
    assert all(isinstance(s, Atoms) for s in structures)

    # All structures have the correct chemical formula (FePt ratio preserved)
    # Mock returns 2 atoms (Fe, Pt) for FePt unit cell.
    # Replicated 2x2x2 = 8 unit cells -> 16 atoms.
    s0 = structures[0]
    symbols = s0.get_chemical_symbols()  # type: ignore[no-untyped-call]
    assert "Fe" in symbols
    assert "Pt" in symbols
    assert len(s0) == 16 # 2 atoms * 8

    # Each structure is slightly different (verify positions)
    # Check first vs second
    pos0 = structures[0].positions  # type: ignore[no-untyped-call]
    pos1 = structures[1].positions  # type: ignore[no-untyped-call]
    assert not np.allclose(pos0, pos1)


def test_uat_03_02_defect_generation() -> None:
    """
    Scenario 03-02: Defect Generation
    Objective: Verify that the system can introduce vacancies.
    """
    # 1. Preparation
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[4, 4, 4],  # Large enough supercell
        policy_name=ExplorationPolicy.DEFECTS,
        vacancy_rate=0.05,
    )
    generator = StructureGenerator(config)

    # 2. Action
    structures = list(generator.generate(n_candidates=1))

    # 3. Expectation
    defect_atoms = structures[0]

    # Get pristine count for comparison
    pristine_config = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    pristine_gen = StructureGenerator(pristine_config)
    pristine_atoms = next(pristine_gen.generate(1))

    # The returned structure has fewer atoms than the pristine bulk
    assert len(defect_atoms) < len(pristine_atoms)

    # The lattice vectors remain unchanged
    assert np.allclose(defect_atoms.get_cell(), pristine_atoms.get_cell())  # type: ignore[no-untyped-call]
