import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones

from pyacemaker.domain_models.distillation import CutoutConfig
from pyacemaker.utils.extraction import (
    _passivate_surface,
    _pre_relax_buffer,
    extract_intelligent_cluster,
)


@pytest.fixture
def simple_cubic():
    # 5x5x5 simple cubic lattice of pseudo atoms (e.g. Ar)
    atoms = bulk("Ar", "sc", a=2.0)
    atoms = atoms * (5, 5, 5)
    # central atom should be around index 62 for 5x5x5
    # center of the cell is 5.0, 5.0, 5.0
    # atoms.positions range from 0 to 8
    # Let's find the atom closest to center
    center = np.array([5.0, 5.0, 5.0])
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    target_idx = np.argmin(distances)
    return atoms, target_idx


def test_spherical_cutout_logic(simple_cubic):
    atoms, target_idx = simple_cubic

    config = CutoutConfig(
        core_radius=2.1,  # Should include 1st neighbors in SC (a=2.0)
        buffer_radius=1.0,  # Total radius 3.1 -> should include 2nd neighbors
        enable_pre_relaxation=False,
        enable_passivation=False,
    )

    cluster = extract_intelligent_cluster(atoms, [int(target_idx)], config)

    # Assertions
    assert isinstance(cluster, Atoms)
    assert len(cluster) > 1
    assert "force_weight" in cluster.arrays

    force_weights = cluster.get_array("force_weight")

    # Original target atom should be in the core (1.0)
    # In SC lattice, 1st neighbors are at dist=2.0 (6 atoms) -> total 7 core atoms
    core_count = np.sum(force_weights == 1.0)
    assert core_count == 7

    # 2nd neighbors are at dist=2.82 (12 atoms).
    # Total atoms = 1 + 6 + 12 = 19
    assert len(cluster) == 19
    buffer_count = np.sum(force_weights == 0.0)
    assert buffer_count == 12


def test_pre_relax_buffer(simple_cubic):
    atoms, target_idx = simple_cubic

    config = CutoutConfig(
        core_radius=2.1,
        buffer_radius=1.0,
        enable_pre_relaxation=False,  # We call it manually for test
        enable_passivation=False,
    )

    cluster = extract_intelligent_cluster(atoms, [int(target_idx)], config)
    initial_positions = cluster.positions.copy()

    calc = LennardJones(epsilon=1.0, sigma=2.0)

    relaxed_cluster = _pre_relax_buffer(cluster, calc)

    final_positions = relaxed_cluster.positions
    force_weights = relaxed_cluster.get_array("force_weight")

    core_mask = force_weights == 1.0
    buffer_mask = force_weights == 0.0

    # Core atoms should NOT have moved
    assert np.allclose(initial_positions[core_mask], final_positions[core_mask])

    # Buffer atoms SHOULD have moved
    assert not np.allclose(initial_positions[buffer_mask], final_positions[buffer_mask])


def test_auto_passivation():
    # Create a small "broken" silicon cluster to test passivation
    # Just 2 Si atoms far apart to simulate dangling bonds
    atoms = Atoms("Si2", positions=[[0, 0, 0], [4, 0, 0]])

    # Simulate extraction by manually adding force weights
    # Make one core, one buffer
    atoms.set_array("force_weight", np.array([1.0, 0.0]))

    config = CutoutConfig(passivation_element="H")

    # _passivate_surface should add H atoms to the buffer atom
    passivated_cluster = _passivate_surface(atoms, "H")

    # It should have added at least one H atom
    assert len(passivated_cluster) > 2

    # Check that new atoms are H
    added_atoms = passivated_cluster[2:]
    assert all(a.symbol == "H" for a in added_atoms)

    # Check force_weights of new atoms
    force_weights = passivated_cluster.get_array("force_weight")
    assert all(w == 0.0 for w in force_weights[2:])

    # Check distance
    # Buffer atom was Si at index 1
    # Standard Si-H distance ~ 1.48 A (from covalent radii)
    buffer_pos = passivated_cluster.positions[1]
    for h_pos in passivated_cluster.positions[2:]:
        dist = np.linalg.norm(h_pos - buffer_pos)
        assert 1.0 < dist < 2.0  # Physically reasonable bond distance


def test_extract_intelligent_cluster_empty_targets():
    atoms = Atoms("Ar")
    config = CutoutConfig()
    with pytest.raises(ValueError, match="target_atoms list cannot be empty."):
        extract_intelligent_cluster(atoms, [], config)
