import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.embedding import embed_cluster


def extract_intelligent_cluster(
    structure: Atoms, target_atoms: list[int], config: CutoutConfig
) -> Atoms:
    """
    Extracts a local cluster around a set of target atoms from a structure,
    applies force weights (Core=1.0, Buffer=0.0), optionally passivates with dummy atoms,
    and embeds into a periodic box with a vacuum layer.

    Args:
        structure: The source atomic structure (usually periodic).
        target_atoms: The indices of the epicentre atoms.
        config: Cutout configuration including core_radius and buffer_radius.

    Returns:
        Atoms: The embedded cluster with 'force_weight' array in arrays, passivated.
    """
    if not target_atoms:
        msg = "target_atoms list cannot be empty."
        raise ValueError(msg)

    config = CutoutConfig.model_validate(config)

    total_cutoff = config.core_radius + config.buffer_radius

    # Use ASE's neighbor_list to find neighbors respecting PBC
    i_indices, j_indices, D_vectors = neighbor_list("ijD", structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]

    # Use the first target atom as the visual center
    # In a full multi-center implementation, we would merge spheres.
    # For now, simplify by centering on the first target atom, but including neighbors from all.
    center_index = target_atoms[0]

    # Find neighbors for ANY target atom
    mask = np.isin(i_indices, target_atoms)
    neighbors_indices = j_indices[mask]
    vectors = D_vectors[mask]  # vectors from i to j

    # We want unique neighbor atoms. We'll find their coordinates relative to the center_index atom
    # However, `vectors` here are from *their respective target atom i*.
    # So we calculate the absolute positions of neighbors by unwrapping them.

    # Simpler approach: unwrapped position of j = position of i + D_vector
    orig_pos = structure.get_positions()  # type: ignore[no-untyped-call]
    i_pos = orig_pos[i_indices[mask]]
    unwrapped_j_pos = i_pos + vectors

    # We need to collect the target atoms themselves.
    # Also, we might have multiple paths to the same j atom if spheres overlap.
    # We should average their positions or just take the first one found (unwrapping is consistent).
    unique_j_indices, unique_idx = np.unique(neighbors_indices, return_index=True)
    j_unwrapped = unwrapped_j_pos[unique_idx]

    # Now we have all unique neighbor atoms. We also need to include the target atoms that might not be in the neighbor list.
    all_indices = np.union1d(target_atoms, unique_j_indices)

    # Map index to unwrapped position
    idx_to_pos = {}
    for i in target_atoms:
        idx_to_pos[i] = orig_pos[i] # Initial positions for targets

    for j_idx, pos in zip(unique_j_indices, j_unwrapped, strict=False):
        # We might overwrite targets with unwrapped versions if they are neighbors of other targets
        idx_to_pos[j_idx] = pos

    # Determine core vs buffer.
    # An atom is core if it is a target atom, OR if it's within core_radius of ANY target atom.

    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]

    cluster_positions = []
    cluster_symbols = []
    cluster_weights = []
    cluster_indices = []

    for idx in all_indices:
        pos = idx_to_pos[idx]
        cluster_positions.append(pos)
        cluster_symbols.append(all_symbols[idx])
        cluster_indices.append(idx)

        # Determine weight
        if idx in target_atoms:
            cluster_weights.append(1.0)
        else:
            # Check distance to nearest target atom
            # Reconstruct vector from any target atom
            # (Note: this is an approximation since we unwrapped relative to one specific i, but fine for local clusters)
            min_dist = np.inf
            for t_idx in target_atoms:
                dist = np.linalg.norm(pos - idx_to_pos[t_idx])
                min_dist = min(min_dist, dist)

            if min_dist <= config.core_radius + 1e-6:
                cluster_weights.append(1.0)
            else:
                cluster_weights.append(0.0)

    # Convert to arrays
    final_positions = np.array(cluster_positions)
    # Shift center to origin
    center_pos = idx_to_pos[center_index]
    final_positions -= center_pos

    cluster = Atoms(symbols=cluster_symbols, positions=final_positions, pbc=False)
    cluster.new_array("force_weight", np.array(cluster_weights))  # type: ignore[no-untyped-call]
    cluster.new_array("original_index", np.array(cluster_indices))  # type: ignore[no-untyped-call]

    # Passivation logic
    if config.enable_passivation:
        cluster = _passivate_surface(cluster, config.passivation_element)

    # Boundary pre-relaxation logic is meant to be handled with MACE,
    # but we just flag it here or handle it in the orchestrator/oracle.
    # The spec mentions "_pre_relax_buffer(cluster, mace_calc)".
    # We'll leave the actual relaxation execution to the Oracle or Manager since we don't have MACE here.

    return embed_cluster(cluster, buffer=5.0)

def _passivate_surface(cluster: Atoms, passivation_element: str) -> Atoms:
    """
    Dummy passivation logic for UAT.
    Finds atoms with weight 0.0 at the outer edge and attaches a passivation atom.
    """
    # Simple heuristic: find buffer atoms (weight 0.0) and add a dummy atom radially outward.
    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    positions = cluster.get_positions()  # type: ignore[no-untyped-call]

    # Center of mass
    com = np.mean(positions, axis=0)

    new_symbols = []
    new_positions = []
    new_weights = []

    for i, w in enumerate(weights):
        if w == 0.0:
            # Add a dummy atom 1.0 Angstrom outward
            vec = positions[i] - com
            norm = np.linalg.norm(vec)
            if norm > 0:
                direction = vec / norm
                new_pos = positions[i] + direction * 1.0
                new_symbols.append(passivation_element)
                new_positions.append(new_pos)
                new_weights.append(0.0) # Passivation atoms are also buffer

    if new_symbols:
        cluster.extend(Atoms(symbols=new_symbols, positions=new_positions))  # type: ignore[no-untyped-call]

        # update weights array
        final_weights = np.concatenate([weights, new_weights])
        cluster.set_array("force_weight", final_weights)  # type: ignore[no-untyped-call]

    return cluster


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
