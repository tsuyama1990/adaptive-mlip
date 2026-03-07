import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.exceptions import GeneratorError
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_cold_start_policy() -> None:
    from pyacemaker.core.policy import ColdStartPolicy
    config = StructureConfig(elements=["Fe"], supercell_size=[1,1,1])
    policy = ColdStartPolicy()
    base = Atoms("Fe")
    structures = list(policy.generate(base, config, 10))
    # Cold start yields 1 structure regardless of n
    assert len(structures) == 1
    assert isinstance(structures[0], Atoms)


def test_rattle_policy() -> None:
    from pyacemaker.core.policy import RattlePolicy
    config = StructureConfig(elements=["Fe"], supercell_size=[1,1,1], rattle_stdev=0.1)
    policy = RattlePolicy()
    base = Atoms("Fe", positions=[[0,0,0]])
    structures = list(policy.generate(base, config, 5))
    assert len(structures) == 5
    assert not np.allclose(structures[0].positions, base.positions)


def test_defect_policy() -> None:
    from pyacemaker.core.policy import DefectPolicy
    config = StructureConfig(elements=["Fe"], supercell_size=[2,2,2])
    policy = DefectPolicy()
    base = Atoms("Fe2", positions=[[0,0,0], [1,1,1]])
    structures = list(policy.generate(base, config, 2))
    assert len(structures) == 2
    assert len(structures[0]) == 1 # 1 atom removed from 2


def test_strain_policy() -> None:
    from pyacemaker.core.policy import StrainPolicy
    config = StructureConfig(elements=["Fe"], supercell_size=[1,1,1], strain_mode="volume", strain_magnitude=0.05)
    policy = StrainPolicy()
    base = Atoms("Fe", cell=[1,1,1], pbc=True)
    structures = list(policy.generate(base, config, 2))
    assert len(structures) == 2
    assert structures[0].get_volume() != base.get_volume() # type: ignore[no-untyped-call]


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

    # Updated error message expectation
    with pytest.raises(GeneratorError, match="Base generator failed"):
        next(generator.generate(1))


def test_generate_local() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        local_generation_strategy="random_displacement",
        rattle_stdev=0.1
    )
    generator = StructureGenerator(config)

    # Create dummy base structure
    base = Atoms("Fe2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)

    candidates = list(generator.generate_local(base, n_candidates=5))

    assert len(candidates) == 5
    for c in candidates:
        assert len(c) == 2
        assert not np.allclose(c.positions, base.positions)
        assert np.allclose(c.cell, base.cell)
