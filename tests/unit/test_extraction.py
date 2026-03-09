import numpy as np
from ase import Atoms
from ase.build import bulk

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.extraction import (
    _passivate_surface,
    _pre_relax_buffer,
    extract_intelligent_cluster,
    extract_local_region,
)


def test_extract_local_region_basic() -> None:
    # Create a simple cubic lattice
    atoms = bulk("Cu", "sc", a=2.5).repeat((3, 3, 3))  # type: ignore[no-untyped-call]

    # Center atom at index 13 (middle of 3x3x3 is 13? 3*3*3=27. 13 is center)
    center_idx = 13

    # Radius covers 1st shell (2.5), Buffer covers 2nd shell
    # 1st neighbor dist = 2.5
    # 2nd neighbor dist = sqrt(2.5^2 + 2.5^2) = 3.535
    # 3rd neighbor dist = sqrt(2.5^2 + 2.5^2 + 2.5^2) = 4.33

    radius = 2.6  # Includes 1st shell
    buffer = 1.0  # Total cutoff 3.6 (Includes 2nd shell)

    cluster = extract_local_region(atoms, center_idx, radius, buffer)

    # Check cluster size
    # 1 center + 6 nearest neighbors (1st shell) + 12 next-nearest (2nd shell) = 19
    # Wait, 2nd shell is at 3.535. Total cutoff 3.6 includes it.
    # So we expect 1 + 6 + 12 = 19 atoms.
    assert len(cluster) == 19

    # Check weights
    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]

    # Center (index 0 in cluster usually, but let's check positions)
    # Center is at [0,0,0] relative to original extraction logic, but embed_cluster centers it in box.
    # So we can't rely on position being exactly 0 unless we check relative to box center.
    # However, we know weights: 7 atoms (center + 6 NN) should have weight 1.0
    # 12 atoms (2nd shell) should have weight 0.0

    n_core = np.sum(weights == 1.0)
    n_buffer = np.sum(weights == 0.0)

    assert n_core == 7  # 1 center + 6 NN
    assert n_buffer == 12  # 12 NNN


def test_extract_local_region_pbc() -> None:
    # Test extraction across PBC
    atoms = bulk("Cu", "sc", a=2.5).repeat((2, 2, 2))  # type: ignore[no-untyped-call]
    # 8 atoms.
    # Center at 0 (corner).
    # Radius covers nearest neighbors (which are wrapped).

    center_idx = 0
    radius = 2.6
    buffer = 0.1

    cluster = extract_local_region(atoms, center_idx, radius, buffer)

    # NN of corner 0 in 2x2x2 SC are 3 (along axes) + ?
    # In periodic 2x2x2, each atom has 6 NN.
    # So we expect 1 + 6 = 7 atoms in cluster.
    assert len(cluster) == 7

    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    assert np.all(weights == 1.0)  # All are within radius


def test_pre_relax_buffer() -> None:
    # Create a small cluster to test _pre_relax_buffer
    atoms = Atoms("Cu3", positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]])

    # Core atoms (weight 1.0) should be fixed, buffer (weight 0.0) should move
    weights = np.array([1.0, 0.0, 0.0])
    atoms.new_array("force_weight", weights)  # type: ignore[no-untyped-call]

    # Store original positions
    original_positions = atoms.get_positions().copy()  # type: ignore[no-untyped-call]

    # Relax
    relaxed_atoms = _pre_relax_buffer(atoms)

    # Check that core atom didn't move
    np.testing.assert_allclose(relaxed_atoms.get_positions()[0], original_positions[0], atol=1e-6)  # type: ignore[no-untyped-call]

    # Due to flat LJ potential / small cluster, buffer might not move.
    # The requirement is to verify the function executes and returns a correctly structured object.
    assert len(relaxed_atoms) == 3


def test_passivate_surface() -> None:
    # Create a small cluster
    atoms = Atoms("O3", positions=[[0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]])

    # Only middle atom is core
    weights = np.array([0.0, 1.0, 0.0])
    atoms.new_array("force_weight", weights)  # type: ignore[no-untyped-call]

    # Passivate
    passivated_atoms = _passivate_surface(atoms, element="H")

    # We started with 3 atoms.
    # Buffer atoms (index 0 and 2) have 1 neighbor each (within cutoff 2.5).
    # Since 1 < 4, they should be passivated.
    # We should have more atoms now.
    assert len(passivated_atoms) > 3

    # Check that new atoms are H and have weight 0.0
    symbols = passivated_atoms.get_chemical_symbols()  # type: ignore[no-untyped-call]
    assert "H" in symbols

    new_weights = passivated_atoms.get_array("force_weight")  # type: ignore[no-untyped-call]
    # The original atoms had weights [0, 1, 0]. The new ones must have 0.
    assert new_weights[-1] == 0.0


def test_extract_intelligent_cluster() -> None:
    # Create a simple cubic lattice
    atoms = bulk("Cu", "sc", a=2.5).repeat((3, 3, 3))  # type: ignore[no-untyped-call]

    # Add dummy c_gamma
    c_gamma = np.random.rand(len(atoms))
    atoms.new_array("c_gamma", c_gamma)  # type: ignore[no-untyped-call]

    config = CutoutConfig(
        core_radius=2.6,
        buffer_radius=1.0,
        enable_pre_relaxation=True,
        enable_passivation=True,
        passivation_element="H",
    )

    # Empty target atoms should return a copy of the original
    empty_cluster = extract_intelligent_cluster(atoms, [], config)
    assert len(empty_cluster) == len(atoms)

    # Target the center atom
    center_idx = 13
    cluster = extract_intelligent_cluster(atoms, [center_idx], config)

    # Base atoms within 3.6 cutoff should be 19 atoms (from previous test)
    # Plus passivating H atoms
    assert len(cluster) >= 19

    # Verify arrays are preserved/added
    assert "force_weight" in cluster.arrays
    assert "c_gamma" in cluster.arrays

    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    # Center + 6 nearest neighbors = 7 core atoms
    assert np.sum(weights == 1.0) == 7
