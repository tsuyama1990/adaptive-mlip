import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from pyacemaker.domain_models.constants import EMBEDDING_TOLERANCE_CELL


def embed_cluster(cluster: Atoms, buffer: float, copy: bool = True) -> Atoms:
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
        copy:   If True (default), a new Atoms object is returned (safer).
                If False, the input cluster is modified in-place (faster, memory efficient).

    Returns:
        Atoms object with periodic boundary conditions set to True and
        positions centered in the new cell.

    Raises:
        ValueError: If the input cluster is empty (contains no atoms).
    """
    if len(cluster) == 0:
        msg = "Cannot embed empty cluster"
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

    # Calculate shift
    center_of_box = cell_lengths / 2.0
    center_of_atoms = (min_xyz + max_xyz) / 2.0
    shift = center_of_box - center_of_atoms

    # Handle object creation vs modification
    if copy:
        # Create a deep copy of the Atoms object to ensure positions are not shared
        target = cluster.copy()  # type: ignore[no-untyped-call]
        target.positions = target.positions.copy()
    else:
        # In-place modification of the original object: absolutely no new copies
        target = cluster

    # Modify the target (whether it's the copy or original)
    target.set_cell(cell_lengths)
    target.set_pbc(True)
    target.positions += shift

    # Explicit cast to Atoms to satisfy type checker if ASE doesn't return strictly Atoms
    # or just return as is, assuming target is Atoms.
    # The ignore was because mypy thinks ASE methods might return something else?
    # Actually ASE copy() returns 'Atoms' (or subclasses).
    return target  # type: ignore[no-any-return]
