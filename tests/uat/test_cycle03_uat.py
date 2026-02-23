import numpy as np

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_uat_03_01_generate_candidates() -> None:
    """
    Scenario 03-01: Verify generation of perturbed structures (Rattle).
    """
    # Preparation
    config = StructureConfig(
        elements=["Fe", "Pt"],
        supercell_size=[1, 1, 1],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
    )

    # Action
    generator = StructureGenerator(config)
    candidates = list(generator.generate(n_candidates=10))

    # Expectation
    assert len(candidates) == 10

    formulas = [c.get_chemical_formula() for c in candidates]
    # Check consistency
    first_formula = formulas[0]
    assert all(f == first_formula for f in formulas)

    # Check diversity
    # Compare positions of first two
    pos0 = candidates[0].get_positions()
    pos1 = candidates[1].get_positions()
    assert not np.allclose(pos0, pos1)


def test_uat_03_02_defect_generation() -> None:
    """
    Scenario 03-02: Verify generation of structures with defects.
    """
    # Preparation
    # Use a larger supercell to ensure we have enough atoms to remove
    config = StructureConfig(
        elements=["Fe", "Pt"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.DEFECTS,
        vacancy_rate=0.1,  # Increased slightly to ensure removal on small mock systems
    )

    # Action
    generator = StructureGenerator(config)
    candidates = list(generator.generate(n_candidates=1))
    defective_atoms = candidates[0]

    # Expectation
    # To verify reduction, we generate a reference "Cold Start" structure
    # Pydantic v2 uses model_copy()
    config_base = config.model_copy(update={"policy_name": ExplorationPolicy.COLD_START})
    gen_base = StructureGenerator(config_base)
    base_candidates = list(gen_base.generate(n_candidates=1))
    base_atoms = base_candidates[0]

    assert len(defective_atoms) < len(base_atoms)

    # Check cell preservation
    assert np.allclose(defective_atoms.get_cell(), base_atoms.get_cell())
