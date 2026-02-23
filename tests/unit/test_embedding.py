import numpy as np
import pytest
from ase import Atoms

from pyacemaker.utils.embedding import embed_cluster


def test_embed_cluster_single_atom() -> None:
    """Test embedding a single atom."""
    atom = Atoms("H", positions=[[0, 0, 0]])
    buffer = 5.0
    embedded = embed_cluster(atom, buffer)

    # For single atom, min=max=0. Dims=0. Cell = 0+buffer = 5.
    assert np.allclose(embedded.cell.lengths(), [buffer, buffer, buffer])
    assert np.all(embedded.pbc)

    # Atom should be centered
    assert np.allclose(embedded.positions[0], [buffer / 2, buffer / 2, buffer / 2])


def test_embed_cluster_dimer() -> None:
    """Test embedding a dimer."""
    # H2 molecule along x axis, bond length 0.74
    dimer = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    buffer = 10.0
    embedded = embed_cluster(dimer, buffer)

    # Widths: x=0.74, y=0, z=0
    # Cell: x=10.74, y=10, z=10
    expected_lengths = [0.74 + buffer, buffer, buffer]
    assert np.allclose(embedded.cell.lengths(), expected_lengths)

    # Relative position should be same
    dist = np.linalg.norm(embedded.positions[1] - embedded.positions[0])
    assert np.isclose(dist, 0.74)

    # Check centering
    center = embedded.cell.lengths() / 2.0
    com = np.mean(embedded.positions, axis=0)
    assert np.allclose(center, com)


def test_embed_cluster_empty() -> None:
    """Test that embedding an empty cluster raises ValueError."""
    empty = Atoms()
    with pytest.raises(ValueError, match="Cannot embed empty cluster"):
        embed_cluster(empty, 10.0)
