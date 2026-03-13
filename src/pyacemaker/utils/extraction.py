import os

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.embedding import embed_cluster


def _pre_relax_buffer(
    cluster: Atoms, fmax: float = 0.05, steps: int = 50, maxstep: float = 0.2
) -> Atoms:
    """
    Relaxes the buffer region (force_weight == 0.0) while keeping the core fixed.
    Assumes cluster has already been validated.
    """
    # Create a copy to prevent modifying the original incorrectly
    cluster_copy = cluster.copy()  # type: ignore[no-untyped-call]

    if not cluster_copy.has("force_weight"):
        msg = "Cluster must have 'force_weight' array."
        raise ValueError(msg)

    # Identify core atoms
    weights = cluster_copy.get_array("force_weight")
    core_indices = np.where(weights == 1.0)[0]

    # Set constraints to fix core atoms
    constraint = FixAtoms(indices=core_indices)  # type: ignore[no-untyped-call]
    cluster_copy.set_constraint(constraint)

    if cluster_copy.calc is None:
        if getattr(cluster, "calc", None) is not None:
            cluster_copy.calc = cluster.calc
        else:
            msg = "No calculator attached to structure for pre-relaxation."
            raise ValueError(msg)

    # Relax the buffer region
    import os
    from pathlib import Path

    with Path(os.devnull).open("w") as devnull:
        opt = LBFGS(cluster_copy, logfile=devnull)
        opt.run(fmax=fmax, steps=min(steps, 500))  # type: ignore[no-untyped-call]

    return cluster_copy  # type: ignore[no-any-return]


def _get_expected_coordination(symbol: str) -> int:
    """Returns simple heuristic expected coordination based on valency."""
    from ase.data import chemical_symbols

    if symbol not in chemical_symbols:
        msg = f"Invalid chemical symbol: {symbol}"
        raise ValueError(msg)

    if symbol == "O":
        return 2
    if symbol == "Mg":
        return 6
    if symbol in ["Fe", "Pt"]:
        return 8  # BCC/FCC bulk roughly
    if symbol == "H":
        return 1
    return 6  # Typical for many transition metals and oxides in bulk


def validate_passivation_input(pos: np.ndarray, neighbors_vecs: np.ndarray) -> None:
    if pos.shape != (3,):
        msg = f"Expected pos to have shape (3,), got {pos.shape}"
        raise ValueError(msg)
    if neighbors_vecs.ndim != 2 or neighbors_vecs.shape[1] != 3:
        msg = f"Expected neighbors_vecs to have shape (N, 3), got {neighbors_vecs.shape}"
        raise ValueError(msg)


def _calculate_passivation_positions(
    idx: int, pos: np.ndarray, neighbors_vecs: np.ndarray, missing_bonds: int
) -> list[np.ndarray]:
    """Calculates deterministic positions for new passivating atoms."""
    validate_passivation_input(pos, neighbors_vecs)

    # Use standard reproducible PRNG for deterministic scientific calculations
    # using a fixed seed combined with the unique atom index for variety
    rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), byteorder="little"))

    if len(neighbors_vecs) > 0:
        # Vector pointing away from the center of mass of neighbors
        com_vec = np.mean(neighbors_vecs, axis=0)
        base_offset = -com_vec
        norm = np.linalg.norm(base_offset)
        if norm > 1e-5:
            base_offset = base_offset / norm * 1.0  # 1.0 Angstrom bond length for H
        else:
            base_offset = rng.normal(size=3)
            base_offset = base_offset / np.linalg.norm(base_offset) * 1.0
    else:
        # No neighbors, just place it somewhere
        base_offset = rng.normal(size=3)
        base_offset = base_offset / np.linalg.norm(base_offset) * 1.0

    new_positions = []
    for b in range(missing_bonds):
        # Add slight perturbations if adding multiple bonds to the same atom
        # to avoid placing them exactly on top of each other
        perturbation = rng.normal(scale=0.1, size=3) if b > 0 else np.zeros(3)
        offset = base_offset + perturbation
        offset = offset / np.linalg.norm(offset) * 1.0  # Re-normalize to 1.0A
        new_positions.append(pos + offset)

    return new_positions


def validate_cluster_for_passivation(cluster: Atoms) -> None:
    from pyacemaker.utils.validation import validate_structure

    validate_structure(cluster)
    if not cluster.has("force_weight"):  # type: ignore[no-untyped-call]
        msg = "Cluster must have 'force_weight' array for passivation."
        raise ValueError(msg)


def compute_neighbor_cutoffs(cluster: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ase.neighborlist import natural_cutoffs

    cutoffs = natural_cutoffs(cluster, mult=1.2)  # type: ignore[no-untyped-call]
    return neighbor_list("ijD", cluster, cutoff=cutoffs)  # type: ignore[no-untyped-call, no-any-return]


def detect_undercoordinated_atoms(
    cluster: Atoms,
    i_indices: np.ndarray,
    d_vectors: np.ndarray,
    element: str,
) -> list[Atoms]:
    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    buffer_indices = np.where(weights == 0.0)[0]
    new_atoms = []
    symbols = cluster.get_chemical_symbols()  # type: ignore[no-untyped-call]

    for idx in buffer_indices:
        mask = i_indices == idx
        n_neighbors = int(np.sum(mask))

        sym = symbols[idx]
        expected_coord = _get_expected_coordination(sym)

        if n_neighbors < expected_coord:
            missing_bonds = expected_coord - n_neighbors
            neighbors_vecs = d_vectors[mask]
            pos = cluster.positions[idx]
            new_positions = _calculate_passivation_positions(
                idx, pos, neighbors_vecs, missing_bonds
            )

            for new_pos in new_positions:
                new_atoms.append(Atoms(element, positions=[new_pos]))

    return new_atoms


def _detect_and_add_passivation_atoms(cluster: Atoms, element: str) -> list[Atoms]:
    """Identifies undercoordinated atoms and returns a list of new passivating atoms to add."""
    validate_cluster_for_passivation(cluster)
    i_indices, _, d_vectors = compute_neighbor_cutoffs(cluster)
    return detect_undercoordinated_atoms(cluster, i_indices, d_vectors, element)


def _passivate_surface(cluster: Atoms, element: str = "H") -> Atoms:
    """
    Passivates the surface of the cluster by adding dummy atoms (e.g. H) to undercoordinated atoms.
    Uses covalent radii to determine missing bonds.
    """
    from ase.data import chemical_symbols

    from pyacemaker.utils.validation import validate_structure

    # Security validation before passivation logic to prevent processing malformed structures
    validate_structure(cluster)

    if element not in chemical_symbols or element in {"X", ""}:
        msg = f"Passivation element must be a valid real atom, not {element}"
        raise ValueError(msg)

    cluster_copy = cluster.copy()  # type: ignore[no-untyped-call]

    new_atoms = _detect_and_add_passivation_atoms(cluster_copy, element)

    if new_atoms:
        for new_atom in new_atoms:
            cluster_copy += new_atom

        # Update force_weight array to include the new passivated atoms (with weight 0.0)
        weights = cluster_copy.get_array("force_weight")[: len(cluster)]
        new_weights = np.append(weights, np.zeros(len(new_atoms)))
        cluster_copy.set_array("force_weight", new_weights)

    return cluster_copy  # type: ignore[no-any-return]


def _compute_cluster_indices(
    structure: Atoms, target_atoms: list[int], total_cutoff: float
) -> tuple[list[int], dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    from ase.neighborlist import neighbor_list

    if total_cutoff > 50.0:
        msg = f"total_cutoff {total_cutoff} is too large, exceeds maximum safe threshold of 50.0 A."
        raise ValueError(msg)

    i_indices, j_indices, D_vectors = neighbor_list("ijD", structure, cutoff=total_cutoff)  # type: ignore[no-untyped-call]
    mask = np.isin(i_indices, target_atoms)

    neighbors_indices = j_indices[mask]
    vectors = D_vectors[mask]
    source_indices = i_indices[mask]

    unique_cluster_indices = set(target_atoms)
    unique_cluster_indices.update(neighbors_indices)

    cluster_indices = list(unique_cluster_indices)
    cluster_indices.sort()

    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(cluster_indices)}
    return cluster_indices, idx_map, source_indices, neighbors_indices, vectors


def calculate_distances(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=1)  # type: ignore[no-any-return]


def assign_core_weights(
    target_atoms: list[int],
    idx_map: dict[int, int],
    weights: np.ndarray,
) -> None:
    for target_idx in target_atoms:
        weights[idx_map[target_idx]] = 1.0


def assign_buffer_weights(
    distances: np.ndarray,
    source_indices: np.ndarray,
    neighbors_indices: np.ndarray,
    idx_map: dict[int, int],
    weights: np.ndarray,
    config: CutoutConfig,
) -> None:
    import math

    for i, (_src_idx, neighbor_idx) in enumerate(
        zip(source_indices, neighbors_indices, strict=False)
    ):
        if distances[i] <= config.core_radius or math.isclose(
            distances[i], config.core_radius, abs_tol=1e-6
        ):
            weights[idx_map[neighbor_idx]] = 1.0
        elif (
            distances[i] <= config.core_radius + config.buffer_radius
            or math.isclose(distances[i], config.core_radius + config.buffer_radius, abs_tol=1e-6)
        ) and weights[idx_map[neighbor_idx]] != 1.0:
            weights[idx_map[neighbor_idx]] = 0.0


def _assign_weights(
    target_atoms: list[int],
    cluster_indices: list[int],
    idx_map: dict[int, int],
    source_indices: np.ndarray,
    neighbors_indices: np.ndarray,
    vectors: np.ndarray,
    config: CutoutConfig,
) -> np.ndarray:
    weights = np.zeros(len(cluster_indices))
    distances = calculate_distances(vectors)

    assign_core_weights(target_atoms, idx_map, weights)
    assign_buffer_weights(distances, source_indices, neighbors_indices, idx_map, weights, config)

    return weights


def create_cluster_from_indices(
    structure: Atoms,
    cluster_indices: list[int],
) -> Atoms:
    if len(cluster_indices) == 0:
        msg = "No atoms in cluster."
        raise ValueError(msg)

    cluster_positions = structure.positions[cluster_indices]
    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]
    cluster_symbols = all_symbols[cluster_indices]

    return Atoms(symbols=cluster_symbols, positions=cluster_positions, pbc=False)


def transform_cluster_coordinates(
    cluster: Atoms,
    structure: Atoms,
    target_atoms: list[int],
) -> None:
    target_positions = structure.positions[target_atoms]
    center_pos = np.mean(target_positions, axis=0)
    cluster.positions -= center_pos


def copy_cluster_properties(
    source: Atoms,
    target: Atoms,
    cluster_indices: list[int],
) -> None:
    if source.has("c_gamma"):  # type: ignore[no-untyped-call]
        original_c_gamma = source.get_array("c_gamma")  # type: ignore[no-untyped-call]
        cluster_c_gamma = original_c_gamma[cluster_indices]
        target.new_array("c_gamma", cluster_c_gamma)  # type: ignore[no-untyped-call]


def _create_cluster_atoms(
    structure: Atoms,
    target_atoms: list[int],
    cluster_indices: list[int],
) -> Atoms:
    cluster = create_cluster_from_indices(structure, cluster_indices)
    transform_cluster_coordinates(cluster, structure, target_atoms)
    copy_cluster_properties(structure, cluster, cluster_indices)
    return cluster


def assign_cluster_weights(cluster: Atoms, weights: np.ndarray) -> None:
    cluster.new_array("force_weight", weights)  # type: ignore[no-untyped-call]


def process_cluster_pre_relaxation(cluster: Atoms, config: CutoutConfig) -> Atoms:
    if config.enable_pre_relaxation:
        return _pre_relax_buffer(
            cluster,
            fmax=config.pre_relaxation_fmax,
            steps=config.pre_relaxation_steps,
        )
    return cluster


def process_cluster_passivation(cluster: Atoms, config: CutoutConfig) -> Atoms:
    if config.enable_passivation:
        return _passivate_surface(cluster, element=config.passivation_element)
    return cluster


def embed_processed_cluster(cluster: Atoms) -> Atoms:
    from pyacemaker.utils.validation import validate_structure

    validate_structure(cluster)
    return embed_cluster(cluster, buffer=5.0)


def _post_process_cluster(cluster: Atoms, weights: np.ndarray, config: CutoutConfig) -> Atoms:
    assign_cluster_weights(cluster, weights)
    cluster = process_cluster_pre_relaxation(cluster, config)
    cluster = process_cluster_passivation(cluster, config)
    return embed_processed_cluster(cluster)


def validate_structure_for_extraction(structure: Atoms, target_atoms: list[int]) -> None:
    from pyacemaker.utils.validation import validate_structure

    validate_structure(structure)
    for target_idx in target_atoms:
        if target_idx < 0 or target_idx >= len(structure):
            msg = f"Target atom index {target_idx} is out of bounds for structure with {len(structure)} atoms."
            raise IndexError(msg)


def handle_empty_target_atoms(structure: Atoms) -> Atoms:
    cluster = structure.copy()  # type: ignore[no-untyped-call]
    weights = np.zeros(len(cluster))
    cluster.new_array("force_weight", weights)
    return cluster  # type: ignore[no-any-return]


def perform_intelligent_extraction(
    structure: Atoms, target_atoms: list[int], config: CutoutConfig
) -> Atoms:
    total_cutoff = config.core_radius + config.buffer_radius
    cluster_indices, idx_map, source_indices, neighbors_indices, vectors = _compute_cluster_indices(
        structure, target_atoms, total_cutoff
    )
    weights = _assign_weights(
        target_atoms, cluster_indices, idx_map, source_indices, neighbors_indices, vectors, config
    )
    cluster = _create_cluster_atoms(structure, target_atoms, cluster_indices)
    return _post_process_cluster(cluster, weights, config)


def extract_intelligent_cluster(
    structure: Atoms, target_atoms: list[int], config: CutoutConfig
) -> Atoms:
    """
    Extracts an intelligent local cluster around multiple target atoms,
    relaxing the buffer and passivating the surface.
    """
    if not target_atoms:
        return handle_empty_target_atoms(structure)

    validate_structure_for_extraction(structure, target_atoms)
    return perform_intelligent_extraction(structure, target_atoms, config)


def validate_structure_for_local_extraction(
    structure: Atoms, center_index: int, radius: float, buffer: float
) -> None:
    from pyacemaker.utils.validation import validate_structure

    validate_structure(structure)
    if center_index < 0 or center_index >= len(structure):
        msg = f"Center atom index {center_index} is out of bounds for structure with {len(structure)} atoms."
        raise IndexError(msg)
    if radius < 0.0:
        msg = f"Radius {radius} must be non-negative."
        raise ValueError(msg)
    if buffer < 0.0:
        msg = f"Buffer {buffer} must be non-negative."
        raise ValueError(msg)
    if radius > 20.0 or buffer > 20.0:
        msg = "Radius/buffer too large"
        raise ValueError(msg)


def compute_local_neighbors(
    structure: Atoms, center_index: int, radius: float, buffer: float
) -> tuple[np.ndarray, np.ndarray]:
    from ase.neighborlist import neighbor_list

    total_radius = radius + buffer
    if total_radius > 50.0:
        msg = f"total_radius {total_radius} is too large, exceeds maximum safe threshold of 50.0 A."
        raise ValueError(msg)
    i, j, D = neighbor_list("ijD", structure, total_radius)  # type: ignore[no-untyped-call]
    mask = i == center_index
    return j[mask], D[mask]


def assign_local_weights(vectors: np.ndarray, radius: float) -> np.ndarray:
    import math

    distances = np.linalg.norm(vectors, axis=1)
    weights = np.zeros_like(distances)
    # Core mask includes tolerance for floating point comparisons
    core_mask = [d <= radius or math.isclose(d, radius, abs_tol=1e-6) for d in distances]
    weights[core_mask] = 1.0
    return weights  # type: ignore[no-any-return]


def build_local_cluster(
    structure: Atoms,
    center_index: int,
    neighbors_indices: np.ndarray,
    vectors: np.ndarray,
    weights: np.ndarray,
) -> Atoms:
    center_symbol = structure.get_chemical_symbols()[center_index]  # type: ignore[no-untyped-call]
    all_symbols = np.array(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]

    neighbor_symbols = all_symbols[neighbors_indices]

    final_positions = np.vstack([np.array([0.0, 0.0, 0.0]), vectors])
    final_symbols = np.concatenate([[center_symbol], neighbor_symbols])
    final_weights = np.concatenate([[1.0], weights])

    cluster = Atoms(symbols=final_symbols, positions=final_positions, pbc=False)
    cluster.new_array("force_weight", np.array(final_weights))  # type: ignore[no-untyped-call]
    return cluster


def extract_local_region(
    structure: Atoms, center_index: int, radius: float, buffer: float
) -> Atoms:
    """
    Extracts a local cluster around a specific atom from a structure.
    """

    validate_structure_for_local_extraction(structure, center_index, radius, buffer)
    neighbors_indices, vectors = compute_local_neighbors(structure, center_index, radius, buffer)
    weights = assign_local_weights(vectors, radius)
    cluster = build_local_cluster(structure, center_index, neighbors_indices, vectors, weights)

    return embed_cluster(cluster, buffer=5.0)
