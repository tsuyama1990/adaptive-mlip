import numpy as np
from ase.build import bulk

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.extraction import extract_intelligent_cluster


def test_extract_intelligent_cluster_basic() -> None:
    # Create a simple cubic lattice
    atoms = bulk('Cu', 'sc', a=2.5).repeat((3, 3, 3))  # type: ignore[no-untyped-call]

    # Center atom at index 13 (middle of 3x3x3 is 13? 3*3*3=27. 13 is center)
    center_idx = 13

    # Radius covers 1st shell (2.5), Buffer covers 2nd shell
    # 1st neighbor dist = 2.5
    # 2nd neighbor dist = sqrt(2.5^2 + 2.5^2) = 3.535
    # 3rd neighbor dist = sqrt(2.5^2 + 2.5^2 + 2.5^2) = 4.33

    config = CutoutConfig(
        core_radius=2.6,
        buffer_radius=1.0,
        enable_pre_relaxation=False,
        enable_passivation=False
    )

    cluster = extract_intelligent_cluster(atoms, [center_idx], config)

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
    assert n_buffer == 12 # 12 NNN

def test_extract_intelligent_cluster_pbc() -> None:
    # Test extraction across PBC
    atoms = bulk('Cu', 'sc', a=2.5).repeat((2, 2, 2))  # type: ignore[no-untyped-call]
    # 8 atoms.
    # Center at 0 (corner).
    # Radius covers nearest neighbors (which are wrapped).

    center_idx = 0
    config = CutoutConfig(
        core_radius=2.6,
        buffer_radius=0.1,
        enable_pre_relaxation=False,
        enable_passivation=False
    )

    cluster = extract_intelligent_cluster(atoms, [center_idx], config)

    # NN of corner 0 in 2x2x2 SC are 3 (along axes) + ?
    # In periodic 2x2x2, each atom has 6 NN.
    # So we expect 1 + 6 = 7 atoms in cluster.
    assert len(cluster) == 7

    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    assert np.all(weights == 1.0) # All are within radius
