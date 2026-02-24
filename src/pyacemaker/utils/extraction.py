import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from pyacemaker.utils.embedding import embed_cluster


def extract_local_region(
    structure: Atoms,
    center_index: int,
    radius: float,
    buffer: float
) -> Atoms:
    """
    Extracts a local cluster around a specific atom from a structure.

    The cluster includes all atoms within (radius + buffer).
    Atoms within 'radius' are marked with force_weight=1.0 (core).
    Atoms in the buffer region are marked with force_weight=0.0 (mask).

    The cluster is unwrapped (made contiguous) and then embedded in a new periodic box
    with vacuum padding using embed_cluster.

    Args:
        structure: The source atomic structure (usually periodic).
        center_index: The index of the central atom.
        radius: The radius of the core region (Angstrom).
        buffer: The thickness of the buffer region (Angstrom).

    Returns:
        Atoms: The embedded cluster with 'force_weight' array in arrays.
    """
    total_cutoff = radius + buffer

    # Use ASE's neighbor_list to find neighbors respecting PBC
    # returns i (center indices), j (neighbor indices), D (distance vectors)
    # D is vector from atom i to atom j
    i_indices, j_indices, D_vectors = neighbor_list('ijD', structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]

    # Filter for our center atom
    mask = (i_indices == center_index)
    neighbors_indices = j_indices[mask]
    vectors = D_vectors[mask]

    # Check if neighbors found
    # Even if no neighbors (isolated atom), we proceed with center only.

    # Prepare cluster data
    # Center atom at origin (0,0,0)
    center_symbol = structure.get_chemical_symbols()[center_index]  # type: ignore[no-untyped-call]

    # Initialize lists with center atom
    # Lists are faster for appending than numpy arrays
    cluster_positions = [[0.0, 0.0, 0.0]]
    cluster_symbols = [center_symbol]
    cluster_weights = [1.0]  # Center is core

    # We need to map original indices to chemical symbols
    # Fetch symbols once (list)
    all_symbols = structure.get_chemical_symbols()  # type: ignore[no-untyped-call]

    for idx, vec in zip(neighbors_indices, vectors, strict=False):
        dist = np.linalg.norm(vec)

        # Determine weight
        # Core: dist <= radius. Buffer: radius < dist <= total_cutoff
        # We use strict inequality for buffer to avoid floating point issues at boundary.
        # If dist is exactly radius, it's core.
        weight = 1.0 if dist <= radius + 1e-6 else 0.0

        cluster_positions.append(vec.tolist())
        cluster_symbols.append(all_symbols[idx])
        cluster_weights.append(weight)

    # Create Atoms object
    # pbc=False initially, embed_cluster will handle boxing
    cluster = Atoms(
        symbols=cluster_symbols,
        positions=cluster_positions,
        pbc=False
    )

    # Store weights in arrays
    # 'force_weight' is standard for Pacemaker
    cluster.new_array("force_weight", np.array(cluster_weights))  # type: ignore[no-untyped-call]

    # Embed cluster with standard padding
    # This centers the cluster in a box with vacuum padding
    return embed_cluster(cluster, buffer=5.0)
