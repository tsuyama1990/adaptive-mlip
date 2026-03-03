import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from pyacemaker.domain_models.defaults import EMBEDDING_TOLERANCE_CELL


def embed_cluster(cluster: Atoms, buffer: float) -> Atoms:
    """
    Embeds a cluster of atoms into a periodic box with vacuum padding.

    The function calculates the bounding box of the input cluster, adds a specified
    vacuum buffer to each dimension, and centers the cluster within the new cell.
    Periodic boundary conditions (PBC) are enabled for all dimensions.

    This function always returns a new Atoms object to prevent data corruption.

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

    if len(cluster) > 100000:
        msg = f"Cluster size {len(cluster)} exceeds maximum allowed limit (100000 atoms)"
        raise ValueError(msg)

    if buffer <= 0:
        msg = f"Buffer must be positive: {buffer}"
        raise ValueError(msg)

    if buffer > 1000.0:
        msg = f"Buffer is excessively large: {buffer} (limit 1000.0)"
        raise ValueError(msg)

    # Get bounding box (no copy)
    positions: NDArray[np.float64] = cluster.get_positions()  # type: ignore[no-untyped-call]

    # Validation: Ensure positions is valid (redundant if ASE is valid, but good for type safety)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = f"Invalid positions shape: {positions.shape}. Expected (N, 3)."
        raise ValueError(msg)

    min_xyz = np.min(positions, axis=0)
    max_xyz = np.max(positions, axis=0)

    # Calculate dimensions
    dims = max_xyz - min_xyz
    cell_lengths = dims + buffer

    # Validate cell dimensions
    if np.any(cell_lengths <= EMBEDDING_TOLERANCE_CELL):
        msg = f"Resulting cell dimensions must be positive (> {EMBEDDING_TOLERANCE_CELL}): {cell_lengths}. Increase buffer."
        raise ValueError(msg)

    # Validate volume limits to prevent memory exhaustion and OOM crashes
    vol = cell_lengths[0] * cell_lengths[1] * cell_lengths[2]
    if vol > 1e7:  # Arbitrary high limit to prevent malicious crashes
        msg = f"Resulting cell volume {vol} is excessively large. Limit buffer size."
        raise ValueError(msg)

    # Calculate shift
    center_of_box = cell_lengths / 2.0
    center_of_atoms = (min_xyz + max_xyz) / 2.0
    shift = center_of_box - center_of_atoms

    # Pre-calculate new positions using numpy broadcast to prevent intermediate large arrays in Atoms
    new_positions = positions + shift

    # We do NOT use in-place modification to prevent data corruption.
    # Instead, we create a fresh Atoms object efficiently with symbols and calculated arrays.
    symbols = cluster.get_chemical_symbols()  # type: ignore[no-untyped-call]

    new_cluster = Atoms(symbols=symbols, positions=new_positions, cell=cell_lengths, pbc=True)

    # Copy across relevant arrays (like force_weight)
    for name, array in cluster.arrays.items():
        if name not in ["positions", "numbers"]:
            new_cluster.new_array(name, array.copy())  # type: ignore[no-untyped-call]

    return new_cluster
