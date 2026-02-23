import numpy as np
from ase import Atoms


def embed_cluster(cluster: Atoms, buffer: float) -> Atoms:
    """
    Embeds a cluster of atoms into a periodic box with vacuum padding.

    Args:
        cluster: The atomic cluster to embed.
        buffer: The amount of vacuum to add to the bounding box dimensions (in Angstroms).
                This is the total vacuum added to each dimension (or gap between images).

    Returns:
        Atoms object with periodic boundary conditions and centered in the new cell.

    Raises:
        ValueError: If the cluster is empty.
    """
    if len(cluster) == 0:
        msg = "Cannot embed empty cluster"
        raise ValueError(msg)

    # Get bounding box
    positions = cluster.get_positions()
    min_xyz = np.min(positions, axis=0)
    max_xyz = np.max(positions, axis=0)

    # Calculate dimensions
    dims = max_xyz - min_xyz
    cell_lengths = dims + buffer

    # Create new atoms object
    new_cluster = cluster.copy()  # type: ignore[no-untyped-call]
    new_cluster.set_cell(cell_lengths)
    new_cluster.set_pbc(True)

    # Center atoms in the new cell
    center_of_box = cell_lengths / 2.0
    center_of_atoms = (min_xyz + max_xyz) / 2.0
    shift = center_of_box - center_of_atoms

    new_cluster.translate(shift)

    return new_cluster
