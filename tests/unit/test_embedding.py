import numpy as np
import pytest
from ase import Atoms

from pyacemaker.utils.embedding import embed_cluster


def test_embed_cluster_basic() -> None:
    """Test basic embedding functionality."""
    cluster = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    buffer = 2.0

    embedded = embed_cluster(cluster, buffer=buffer)

    # Check cell size: extent is 1.0 (x), 0 (y,z). Buffer adds 2.0 to each side?
    # Spec says: buffer is added to bounding box dimensions.
    # extent x: max(1) - min(0) = 1.0. New L = 1.0 + 2.0 = 3.0.
    # extent y: 0.0. New L = 0.0 + 2.0 = 2.0.
    # extent z: 0.0. New L = 0.0 + 2.0 = 2.0.

    cell = embedded.get_cell()  # type: ignore[no-untyped-call]
    assert np.allclose(cell.lengths(), [3.0, 2.0, 2.0])
    assert np.all(embedded.get_pbc())  # type: ignore[no-untyped-call]


def test_embed_cluster_copy() -> None:
    """Test that copy=True does not modify original cluster."""
    original = Atoms("H", positions=[[0, 0, 0]])
    original_pos = original.get_positions().copy()  # type: ignore[no-untyped-call]

    _ = embed_cluster(original, buffer=5.0, copy=True)

    assert np.allclose(original.get_positions(), original_pos)  # type: ignore[no-untyped-call]


def test_embed_cluster_inplace() -> None:
    """Test that copy=False modifies original cluster."""
    original = Atoms("H", positions=[[0, 0, 0]])

    # With buffer 10, new cell is 10x10x10. Center is 5,5,5.
    # Original center 0,0,0. Shift should be +5,+5,+5.
    embedded = embed_cluster(original, buffer=10.0, copy=False)

    assert embedded is original
    assert np.allclose(original.get_positions(), [[5.0, 5.0, 5.0]])  # type: ignore[no-untyped-call]


def test_embed_cluster_empty() -> None:
    """Test embedding an empty cluster raises error."""
    empty = Atoms()
    with pytest.raises(ValueError, match="Cannot embed empty cluster"):
        embed_cluster(empty, buffer=1.0)


def test_embed_cluster_invalid_buffer() -> None:
    """Test negative buffer raises error."""
    cluster = Atoms("H", positions=[[0, 0, 0]])
    with pytest.raises(ValueError, match="Buffer must be positive"):
        embed_cluster(cluster, buffer=-5.0)


def test_embed_cluster_large() -> None:
    """Test large cluster."""
    positions = np.random.rand(100, 3) * 100.0
    cluster = Atoms("H100", positions=positions)
    embedded = embed_cluster(cluster, buffer=10.0)

    assert len(embedded) == 100
    assert np.all(embedded.get_pbc())  # type: ignore[no-untyped-call]


def test_embed_cluster_boundary_conditions() -> None:
    """Test cluster with atoms at extreme coordinates."""
    # Atoms far from origin
    cluster = Atoms("H2", positions=[[1000.0, 1000.0, 1000.0], [1001.0, 1000.0, 1000.0]])
    buffer = 2.0
    # Expected cell: 1.0 (extent) + 2.0 (buffer) = 3.0 in x. 2.0 in y, z.

    embedded = embed_cluster(cluster, buffer=buffer)
    cell = embedded.get_cell()  # type: ignore[no-untyped-call]

    assert np.allclose(cell.lengths(), [3.0, 2.0, 2.0])

    # Check centering: center of box is 1.5, 1.0, 1.0.
    # Center of atoms was 1000.5, 1000.0, 1000.0.
    # Shift = 1.5 - 1000.5 = -999.0 in x.
    # New pos 1: 1000 - 999 = 1.0.
    # New pos 2: 1001 - 999 = 2.0.

    pos = embedded.get_positions()  # type: ignore[no-untyped-call]
    assert np.allclose(pos[0], [1.0, 1.0, 1.0])
    assert np.allclose(pos[1], [2.0, 1.0, 1.0])
