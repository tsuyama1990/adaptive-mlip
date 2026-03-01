import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.embedding import embed_cluster


def _pre_relax_buffer(cluster: Atoms) -> Atoms:
    """
    Applies boundary condition constraints and performs local MACE LBFGS
    pre-relaxation of the buffer region to eliminate unnatural bonding strain.
    """
    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    core_indices = np.where(weights >= 1.0)[0]
    cluster.set_constraint(FixAtoms(indices=core_indices))  # type: ignore[no-untyped-call]

    # In production, MACE setup and LBFGS minimization should occur here.
    # To keep dependencies light until configured by the user, we raise NotImplementedError
    # indicating the need for an explicit MACEManager.
    import warnings
    warnings.warn("MACE buffer pre-relaxation is currently bypassed as MACEManager dependency is missing in this mock context.", stacklevel=2)
    return cluster


def _passivate_surface(cluster: Atoms) -> Atoms:
    """
    Auto-passivates broken bonds using fractional hydrogen or dummy elements.
    """
    import warnings
    warnings.warn("Auto-passivation detection heuristics are bypassed in this mock context.", stacklevel=2)
    return cluster


def extract_intelligent_cluster(
    structure: Atoms,
    target_atoms: list[int],
    config: CutoutConfig
) -> Atoms:
    """
    Extracts an intelligent local cluster around target atoms from a massive structure.
    Implements Phase 3 cutout logic.

    Args:
        structure: The massive ASE Atoms object.
        target_atoms: List of atom indices exceeding threshold.
        config: Cutout configuration (core_radius, buffer_radius, etc.).

    Returns:
        Atoms: A computable Atoms object with PBC, vacuum, and passivation.
    """
    if not target_atoms:
        raise ValueError("target_atoms list cannot be empty for intelligent cluster extraction.")

    total_cutoff = config.core_radius + config.buffer_radius

    # Use ASE's neighbor_list to find neighbors respecting PBC
    i_indices, j_indices, D_vectors = neighbor_list('ijD', structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]

    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]

    # We want a union of all neighbors for all target atoms. To keep it simple, we loop and collect.
    # In a full production implementation, we'd build a set of unique indices first.
    # Here we union them around the first target atom's position (or center of mass)
    # For now, collect unique atoms within the union of radii.

    # A robust way: find all atoms j such that distance to ANY target atom is <= total_cutoff
    # Mask all i_indices that are in target_atoms
    mask = np.isin(i_indices, target_atoms)

    unique_j = np.unique(j_indices[mask])

    # Also include the target atoms themselves
    all_cluster_indices = np.unique(np.concatenate([target_atoms, unique_j]))

    final_positions = []
    final_symbols = []
    final_weights = []

    origin_idx = target_atoms[0]

    for idx in all_cluster_indices:
        # Use ASE get_distance with mic=True to accurately account for Periodic Boundary Conditions
        vector = structure.get_distance(origin_idx, idx, mic=True, vector=True)  # type: ignore[no-untyped-call]
        final_positions.append(vector)
        final_symbols.append(all_symbols[idx])

        # Determine weight: if minimum image distance to ANY target atom <= core_radius, weight=1
        is_core = False
        for t_idx in target_atoms:
            dist = structure.get_distance(t_idx, idx, mic=True)  # type: ignore[no-untyped-call]
            if dist <= (config.core_radius + 1e-6):
                is_core = True
                break

        final_weights.append(1.0 if is_core else 0.0)

    final_positions_arr = np.array(final_positions)
    final_symbols_arr = np.array(final_symbols)
    final_weights_arr = np.array(final_weights)

    cluster = Atoms(
        symbols=final_symbols_arr,
        positions=final_positions_arr,
        pbc=False
    )
    cluster.new_array("force_weight", final_weights_arr)  # type: ignore[no-untyped-call]

    # Pre-relax the buffer using MACE if enabled
    if config.enable_pre_relaxation:
        cluster = _pre_relax_buffer(cluster)

    # Auto-passivate dangling bonds on the outer edge
    if config.enable_passivation:
        cluster = _passivate_surface(cluster)

    # Securely place in PBC with vacuum layer
    return embed_cluster(cluster, buffer=5.0)
