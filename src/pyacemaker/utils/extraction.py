import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from pyacemaker.utils.embedding import embed_cluster


def extract_local_region(
    structure: Atoms, center_index: int, radius: float, buffer: float
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
    # neighbor_list uses cell lists internally for O(N) efficiency with valid cutoffs (when cutoff << cell size).
    # For very large structures, this is significantly faster than O(N^2) pairwise calculation.
    # returns i (center indices), j (neighbor indices), D (distance vectors)
    # D is vector from atom i to atom j
    i_indices, j_indices, D_vectors = neighbor_list("ijD", structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]

    # Filter for our center atom
    mask = i_indices == center_index
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
    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]

    # Calculate distances efficiently using numpy
    distances = np.linalg.norm(vectors, axis=1)

    # Determine weights using vectorized masking
    # Core: dist <= radius. Buffer: radius < dist <= total_cutoff
    core_mask = distances <= (radius + 1e-6)
    weights = np.zeros_like(distances)
    weights[core_mask] = 1.0
    # Buffer is implicitly 0.0

    # Convert to lists for ASE Atoms constructor (optional but safe)
    # Append neighbors to cluster lists
    # Note: vectors is (N, 3), cluster_positions expects list of lists or (M, 3) array.
    # We can perform list extension or array concatenation.

    # Using array concatenation for efficiency if N is large.
    # We need to construct the final arrays including the center atom.

    # Vectors for neighbors
    neighbor_positions = vectors

    # Symbols for neighbors
    neighbor_symbols = all_symbols[neighbors_indices]

    # Weights for neighbors
    neighbor_weights = weights

    # Combine with center atom
    final_positions = np.vstack([np.array([0.0, 0.0, 0.0]), neighbor_positions])
    final_symbols = np.concatenate([[center_symbol], neighbor_symbols])
    final_weights = np.concatenate([[1.0], neighbor_weights])

    # Assign back to cluster creation variables
    cluster_positions = final_positions  # type: ignore[assignment]
    cluster_symbols = final_symbols  # type: ignore[assignment]
    cluster_weights = final_weights  # type: ignore[assignment]

    # Create Atoms object
    # pbc=False initially, embed_cluster will handle boxing
    cluster = Atoms(symbols=cluster_symbols, positions=cluster_positions, pbc=False)

    # Store weights in arrays
    # 'force_weight' is standard for Pacemaker
    cluster.new_array("force_weight", np.array(cluster_weights))  # type: ignore[no-untyped-call]

    # Embed cluster with standard padding
    # This centers the cluster in a box with vacuum padding
    return embed_cluster(cluster, buffer=5.0)
