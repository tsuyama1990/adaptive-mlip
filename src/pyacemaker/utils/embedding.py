import numpy as np
from ase import Atoms
from numpy.typing import NDArray


def embed_cluster(cluster: Atoms, buffer: float) -> Atoms:
    """
    Embeds a cluster of atoms into a periodic box with vacuum padding.

    The function calculates the bounding box of the input cluster, adds a specified
    vacuum buffer to each dimension, and centers the cluster within the new cell.
    Periodic boundary conditions (PBC) are enabled for all dimensions.

    Args:
        cluster: The atomic cluster to embed. Must contain at least one atom.
        buffer: The amount of vacuum to add to the bounding box dimensions (in Angstroms).
                This value is added to the extent of the cluster in each dimension.
                For example, if the cluster spans 5.0 A along x and buffer is 10.0 A,
                the new cell length along x will be 15.0 A.

    Returns:
        Atoms object with periodic boundary conditions set to True and
        positions centered in the new cell.

    Raises:
        ValueError: If the input cluster is empty (contains no atoms).
    """
    if len(cluster) == 0:
        msg = "Cannot embed empty cluster"
        raise ValueError(msg)

    # Get bounding box
    positions: NDArray[np.float64] = cluster.get_positions()

    # Validation: Ensure positions is valid (redundant if ASE is valid, but good for type safety)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = f"Invalid positions shape: {positions.shape}. Expected (N, 3)."
        raise ValueError(msg)

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
