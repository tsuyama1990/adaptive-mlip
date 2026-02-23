import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.policy import ColdStartPolicy, DefectPolicy, RattlePolicy, StrainPolicy
from pyacemaker.domain_models.structure import StructureConfig


@pytest.fixture
def base_structure() -> Atoms:
    return Atoms("Fe", positions=[[0, 0, 0]], cell=[2.0, 2.0, 2.0], pbc=True)


def test_cold_start_policy_generates_structure(base_structure: Atoms) -> None:
    config = StructureConfig(elements=["Fe"], supercell_size=[1, 1, 1])
    policy = ColdStartPolicy(config)
    result = policy.apply(base_structure)

    # ColdStartPolicy generates a new stable structure using M3GNet (mocked)
    # It ignores the input base_structure
    assert isinstance(result, Atoms)
    assert "Fe" in result.get_chemical_formula()
    # The generated structure (from bulk) is likely different from our dummy base
    assert result != base_structure


def test_rattle_policy_perturbs_positions(base_structure: Atoms) -> None:
    config = StructureConfig(elements=["Fe"], supercell_size=[1, 1, 1], rattle_stdev=0.1)
    policy = RattlePolicy(config)
    result = policy.apply(base_structure)

    assert len(result) == len(base_structure)
    assert not np.allclose(result.get_positions(), base_structure.get_positions())
    assert np.allclose(result.get_cell(), base_structure.get_cell())


def test_strain_policy_perturbs_cell(base_structure: Atoms) -> None:
    config = StructureConfig(
        elements=["Fe"], supercell_size=[1, 1, 1], strain_range=0.1, strain_mode="full"
    )
    policy = StrainPolicy(config)
    result = policy.apply(base_structure)

    assert len(result) == len(base_structure)
    # Positions (fractional) might be same, but cartesian change.
    # Cell definitely changes.
    assert not np.allclose(result.get_cell(), base_structure.get_cell())


def test_defect_policy_removes_atoms(base_structure: Atoms) -> None:
    # Need more atoms to remove one reliably
    atoms = base_structure * (2, 2, 2)
    config = StructureConfig(elements=["Fe"], supercell_size=[2, 2, 2], vacancy_rate=0.2)
    policy = DefectPolicy(config)
    result = policy.apply(atoms)

    assert len(result) < len(atoms)
    assert np.allclose(result.get_cell(), atoms.get_cell())
