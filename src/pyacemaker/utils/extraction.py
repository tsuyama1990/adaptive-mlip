import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.embedding import embed_cluster


def _pre_relax_buffer(cluster: Atoms) -> Atoms:
    """Mock implementation for pre-relaxing buffer using MACE."""
    # In a full implementation, this would use MACEManager or ASE's MACE calculator
    # and ASE optimizers to relax the buffer while freezing the core.
    # We freeze atoms with force_weight >= 1.0
    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    core_indices = np.where(weights >= 1.0)[0]
    cluster.set_constraint(FixAtoms(indices=core_indices))  # type: ignore[no-untyped-call]
    # In mock, we just return the constrained structure
    return cluster


def _passivate_surface(cluster: Atoms) -> Atoms:
    """Mock implementation for auto-passivating broken bonds."""
    # In a full implementation, this would detect dangling bonds based on
    # electronegativity and distances, and attach dummy atoms (like H).
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
        return structure.copy()  # type: ignore[no-untyped-call, no-any-return]

    center_index = target_atoms[0]
    total_cutoff = config.core_radius + config.buffer_radius

    # Use ASE's neighbor_list to find neighbors respecting PBC
    i_indices, j_indices, D_vectors = neighbor_list('ijD', structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]

    mask = (i_indices == center_index)
    neighbors_indices = j_indices[mask]
    vectors = D_vectors[mask]

    center_symbol = structure.get_chemical_symbols()[center_index]  # type: ignore[no-untyped-call]
    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]

    distances = np.linalg.norm(vectors, axis=1)

    core_mask = distances <= (config.core_radius + 1e-6)
    weights = np.zeros_like(distances)
    weights[core_mask] = 1.0

    neighbor_positions = vectors
    neighbor_symbols = all_symbols[neighbors_indices]
    neighbor_weights = weights

    final_positions = np.vstack([np.array([0.0, 0.0, 0.0]), neighbor_positions])
    final_symbols = np.concatenate([[center_symbol], neighbor_symbols])
    final_weights = np.concatenate([[1.0], neighbor_weights])

    cluster = Atoms(
        symbols=final_symbols,
        positions=final_positions,
        pbc=False
    )
    cluster.new_array("force_weight", np.array(final_weights))  # type: ignore[no-untyped-call]

    # Pre-relax the buffer using MACE if enabled
    if config.enable_pre_relaxation:
        cluster = _pre_relax_buffer(cluster)

    # Auto-passivate dangling bonds on the outer edge
    if config.enable_passivation:
        cluster = _passivate_surface(cluster)

    # Securely place in PBC with vacuum layer
    return embed_cluster(cluster, buffer=5.0)
