import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.embedding import embed_cluster


def _pre_relax_buffer(cluster: Atoms, fmax: float = 0.05, steps: int = 50) -> Atoms:
    """
    Relaxes the buffer region (force_weight == 0.0) while keeping the core fixed.
    """
    # Create a copy to prevent modifying the original incorrectly
    cluster_copy = cluster.copy()  # type: ignore[no-untyped-call]

    # Identify core atoms
    weights = cluster_copy.get_array("force_weight")
    core_indices = np.where(weights == 1.0)[0]

    # Set constraints to fix core atoms
    constraint = FixAtoms(indices=core_indices)  # type: ignore[no-untyped-call]
    cluster_copy.set_constraint(constraint)

    # Apply a mock calculator for the relaxation if one is not attached
    if cluster_copy.calc is None:
        from ase.calculators.lj import LennardJones

        cluster_copy.calc = LennardJones()  # type: ignore[no-untyped-call]

    # Relax the buffer region
    import os
    from pathlib import Path

    with Path(os.devnull).open("w") as devnull:
        opt = LBFGS(cluster_copy, logfile=devnull)
        opt.run(fmax=fmax, steps=steps)  # type: ignore[no-untyped-call]

    return cluster_copy  # type: ignore[no-any-return]


def _passivate_surface(cluster: Atoms, element: str = "H") -> Atoms:
    """
    Passivates the surface of the cluster by adding dummy atoms (e.g. H) to undercoordinated atoms.
    """
    cluster_copy = cluster.copy()  # type: ignore[no-untyped-call]

    # We will just do a simple distance-based passivation mock implementation
    # Find outer atoms (in the buffer region) that have fewer neighbors
    i_indices, _j_indices = neighbor_list("ij", cluster_copy, cutoff=2.5)  # type: ignore[no-untyped-call]

    weights = cluster_copy.get_array("force_weight")
    buffer_indices = np.where(weights == 0.0)[0]

    new_atoms = []

    # Center of mass calculation for deterministic outward vector
    com = cluster_copy.get_center_of_mass()

    for idx in buffer_indices:
        # Number of neighbors for this atom
        n_neighbors = np.sum(i_indices == idx)
        # Add a passivating element to under-coordinated atoms
        if n_neighbors < 4:
            pos = cluster_copy.positions[idx]

            # Use deterministic direction: outward from center of mass
            direction = pos - com
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                # Fallback to a fixed direction if the atom is exactly at CoM
                direction = np.array([1.0, 0.0, 0.0])
                norm = 1.0

            offset = (direction / norm) * 1.0  # 1.0 Angstrom bond length
            new_pos = pos + offset

            new_atoms.append(Atoms(element, positions=[new_pos]))

    if new_atoms:
        for new_atom in new_atoms:
            cluster_copy += new_atom

        # Update force_weight array to include the new passivated atoms (with weight 0.0)
        new_weights = np.append(weights, np.zeros(len(new_atoms)))
        cluster_copy.set_array("force_weight", new_weights)

    return cluster_copy  # type: ignore[no-any-return]


def extract_intelligent_cluster(
    structure: Atoms, target_atoms: list[int], config: CutoutConfig
) -> Atoms:
    """
    Extracts an intelligent local cluster around multiple target atoms,
    relaxing the buffer and passivating the surface.
    """
    if not target_atoms:
        return structure.copy()  # type: ignore[no-untyped-call, no-any-return]

    total_cutoff = config.core_radius + config.buffer_radius

    # We will compute the distances from all atoms to all target atoms
    # Use ASE's neighbor_list for each target atom

    i_indices, j_indices, D_vectors = neighbor_list("ijD", structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]

    mask = np.isin(i_indices, target_atoms)

    neighbors_indices = j_indices[mask]
    vectors = D_vectors[mask]
    source_indices = i_indices[mask]

    # We need a unique set of atoms to include in the cluster
    unique_cluster_indices = set(target_atoms)
    unique_cluster_indices.update(neighbors_indices)

    cluster_indices = list(unique_cluster_indices)
    cluster_indices.sort()  # Ensure deterministic order

    # Mapping from original structure index to cluster index
    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(cluster_indices)}

    # Now we assign weights
    weights = np.zeros(len(cluster_indices))

    # Core atoms are distance <= config.core_radius from ANY target atom
    distances = np.linalg.norm(vectors, axis=1)

    for target_idx in target_atoms:
        weights[idx_map[target_idx]] = 1.0

    for i, (_src_idx, neighbor_idx) in enumerate(
        zip(source_indices, neighbors_indices, strict=False)
    ):
        if distances[i] <= config.core_radius + 1e-6:
            weights[idx_map[neighbor_idx]] = 1.0

    # Create the cluster atoms
    cluster_positions = structure.positions[cluster_indices]

    # Center the cluster roughly around the mean of target atoms to avoid breaking
    target_positions = structure.positions[target_atoms]
    center_pos = np.mean(target_positions, axis=0)

    cluster_positions = cluster_positions - center_pos

    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]
    cluster_symbols = all_symbols[cluster_indices]

    cluster = Atoms(symbols=cluster_symbols, positions=cluster_positions, pbc=False)

    cluster.new_array("force_weight", weights)  # type: ignore[no-untyped-call]

    if structure.has("c_gamma"):  # type: ignore[no-untyped-call]
        original_c_gamma = structure.get_array("c_gamma")  # type: ignore[no-untyped-call]
        cluster_c_gamma = original_c_gamma[cluster_indices]
        cluster.new_array("c_gamma", cluster_c_gamma)  # type: ignore[no-untyped-call]

    if config.enable_pre_relaxation:
        cluster = _pre_relax_buffer(
            cluster,
            fmax=config.pre_relax_fmax,
            steps=config.pre_relax_steps
        )

    if config.enable_passivation:
        cluster = _passivate_surface(cluster, element=config.passivation_element)

    # Finally, embed the cluster into a cell
    return embed_cluster(cluster, buffer=5.0)


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
