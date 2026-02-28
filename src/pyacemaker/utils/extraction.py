from typing import Any, List

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from pyacemaker.domain_models.config import CutoutConfig
from pyacemaker.utils.embedding import embed_cluster


def _pre_relax_buffer(cluster: Atoms, calculator: Calculator) -> None:
    """Relaxes the buffer region using the provided calculator while fixing core atoms."""
    weights = cluster.arrays.get("force_weight", np.ones(len(cluster)))

    # Core atoms have weight == 1.0
    core_indices = np.where(weights == 1.0)[0]

    constraint = FixAtoms(indices=core_indices)  # type: ignore[no-untyped-call]
    cluster.set_constraint(constraint)  # type: ignore[no-untyped-call]

    cluster.calc = calculator
    opt = LBFGS(cluster, logfile=None)
    opt.run(fmax=0.05, steps=50)  # type: ignore[no-untyped-call]

    # Remove constraints and calculator after relaxation to clean up
    cluster.set_constraint()  # type: ignore[no-untyped-call]
    cluster.calc = None


def _passivate_surface(cluster: Atoms, element: str) -> None:
    """Auto-passivates under-coordinated atoms at the boundary."""
    weights = cluster.arrays.get("force_weight", np.ones(len(cluster)))

    # Passivate only atoms in the buffer (weight == 0.0)
    boundary_indices = np.where(weights == 0.0)[0]
    if len(boundary_indices) == 0:
        return

    positions = cluster.positions
    center_of_mass = np.mean(positions, axis=0)

    new_symbols = list(cluster.symbols)
    new_positions = list(positions)
    new_weights = list(weights)

    passivation_distance = 1.0  # approximate bond length for H

    for idx in boundary_indices:
        # Vector from center to atom
        outward_vector = positions[idx] - center_of_mass
        norm = np.linalg.norm(outward_vector)
        if norm > 1e-6:
            direction = outward_vector / norm
            # Add new atom
            new_pos = positions[idx] + direction * passivation_distance
            new_symbols.append(element)
            new_positions.append(new_pos)
            new_weights.append(0.0)  # Passivation atoms get weight 0.0

    # Update cluster by extending
    for i in range(len(cluster), len(new_symbols)):
        cluster.append(Atoms(symbols=[new_symbols[i]], positions=[new_positions[i]])[0])  # type: ignore[no-untyped-call]

    cluster.arrays["force_weight"] = np.array(new_weights)


def extract_intelligent_cluster(
    structure: Atoms, target_atoms: List[int], config: CutoutConfig, calculator: Any = None
) -> Atoms:
    """
    Extracts an intelligent cluster around a set of target atoms.

    Applies force weights according to core and buffer radii.
    Optionally pre-relaxes the buffer and passivates dangling bonds.
    """
    total_cutoff = config.core_radius + config.buffer_radius

    all_indices_to_include = set()

    # Spherical Cutout: find all atoms within radii of ANY target atom
    for center_idx in target_atoms:
        i_indices, j_indices, D_vectors = neighbor_list('ijD', structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]
        mask = (i_indices == center_idx)
        neighbors_indices = j_indices[mask]

        all_indices_to_include.add(center_idx)
        all_indices_to_include.update(neighbors_indices)

    sorted_indices = sorted(all_indices_to_include)

    if not sorted_indices:
        return Atoms()

    cluster = structure[sorted_indices].copy()  # type: ignore[no-untyped-call]

    # Calculate weights efficiently using KDTree/Neighbor list logic
    # instead of N^2
    target_positions = structure.positions[target_atoms]
    cluster_positions = cluster.positions
    weights = np.zeros(len(cluster))

    # For O(N log N) we could use scipy cKDTree, but let's avoid adding new deps
    # and just use vectorized dist calculation per target atom since target_atoms is very small (e.g. 1 defect core).
    for i, pos in enumerate(cluster_positions):
        min_dist = np.min(np.linalg.norm(target_positions - pos, axis=1))
        if min_dist <= config.core_radius + 1e-6:
            weights[i] = 1.0

    cluster.new_array("force_weight", weights)

    # Pre-relaxation
    if config.enable_pre_relaxation and calculator is not None:
        _pre_relax_buffer(cluster, calculator)

    # Auto-Passivation
    if config.enable_passivation:
        _passivate_surface(cluster, config.passivation_element)

    # Finally, embed it
    return embed_cluster(cluster, buffer=5.0)


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
    # neighbor_list uses cell lists internally for O(N) efficiency with valid cutoffs (when cutoff << cell size).
    # For very large structures, this is significantly faster than O(N^2) pairwise calculation.
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
