import numpy as np
from ase import Atoms

from pyacemaker.utils.embedding import embed_cluster


def test_embed_cluster_single_atom() -> None:
    """Test embedding a single atom."""
    atom = Atoms("H", positions=[[0, 0, 0]])
    buffer = 5.0
    embedded = embed_cluster(atom, buffer)

    # Check cell dimensions
    # Single atom at 0,0,0 -> bounding box is 0 size?
    # Actually, embed_cluster should handle this. A single atom has min=max.
    # So width is 0. Cell size should be 0 + buffer? Or buffer on each side?
    # Usually buffer means "vacuum padding". So typically 2*buffer if buffer is on each side.
    # But SPEC says "Add a vacuum buffer (e.g., 10 A)".
    # If the function takes `buffer`, I'll assume it's total vacuum or vacuum on each side.
    # Let's assume vacuum on each side for safety, so cell size = width + 2*buffer.

    # Wait, usually for single atom, cell is just buffer size if buffer is total vacuum?
    # Let's verify what `embedding.py` implementation plan says:
    # "Create a bounding box... Add vacuum buffer... Create orthorhombic cell."

    # If I have one atom at 0, bbox is point. Width 0.
    # If buffer is 10A, cell should be 10A or 20A?
    # Let's implementation define it as `width + buffer`. If width is 0, cell is `buffer`.
    # Let's say buffer is total vacuum added.

    # Let's verify with an implementation:
    # cell_x = (max_x - min_x) + buffer

    assert np.allclose(embedded.cell.lengths(), [buffer, buffer, buffer])
    assert np.all(embedded.pbc)

    # Atom should be centered
    # center = cell / 2 = buffer / 2
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

    # Center of mass check? Or just relative positions maintained?
    # Relative position should be same
    dist = np.linalg.norm(embedded.positions[1] - embedded.positions[0])
    assert np.isclose(dist, 0.74)

    # Centering check
    # Center of cell = [ (0.74+10)/2, 5, 5 ] = [5.37, 5, 5]
    # Center of dimer = [0.37, 0, 0] (in original coords)
    # Shift = [5, 5, 5]
    # Pos0 = [0,0,0] + [5,5,5] = [5,5,5] -> Correct?
    # min_x=0, max_x=0.74. Center_x = 0.37.
    # New center_x = (0.74+10)/2 = 5.37.
    # Shift = 5.37 - 0.37 = 5.0.
    # So positions should be [5,5,5] and [5.74, 5, 5].

    assert np.allclose(embedded.positions[0], [buffer/2, buffer/2, buffer/2])
    assert np.allclose(embedded.positions[1], [buffer/2 + 0.74, buffer/2, buffer/2])
