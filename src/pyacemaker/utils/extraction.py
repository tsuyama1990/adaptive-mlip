import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS

from pyacemaker.domain_models.distillation import CutoutConfig


def extract_intelligent_cluster(
    structure: Atoms,
    target_atoms: list[int],
    config: CutoutConfig,
    calculator: Calculator | None = None,
) -> Atoms:
    """
    Identifies the "epicenter" of uncertainty, spherically extracting a localized cluster,
    and applying force_weight arrays according to CutoutConfig radii.

    Args:
        structure: The full atomic structure.
        target_atoms: List of indices representing the epicenter.
        config: CutoutConfig containing extraction parameters.
        calculator: Optional calculator for pre-relaxation.

    Returns:
        A new Atoms object representing the extracted cluster with force_weights applied.
    """
    if not target_atoms:
        err_msg = "target_atoms list cannot be empty."
        raise ValueError(err_msg)

    # Use neighbor_list to find atoms within core_radius and buffer_radius
    # We find neighbors for all atoms, but we only care about neighbors of target_atoms
    # i is the index of the central atom, j is the index of the neighbor
    i, j, d = neighbor_list("ijd", structure, config.core_radius + config.buffer_radius)  # type: ignore[no-untyped-call]

    # Find all atoms that are within the radii of ANY target atom
    core_mask = np.zeros(len(structure), dtype=bool)
    buffer_mask = np.zeros(len(structure), dtype=bool)

    # Target atoms themselves are in the core
    core_mask[target_atoms] = True

    # Check neighbors of target atoms
    for target_idx in target_atoms:
        # Find neighbors of this target atom
        neighbor_indices = j[i == target_idx]
        neighbor_distances = d[i == target_idx]

        for neighbor_idx, dist in zip(neighbor_indices, neighbor_distances, strict=True):
            if dist <= config.core_radius:
                core_mask[neighbor_idx] = True
            elif dist <= config.core_radius + config.buffer_radius:
                buffer_mask[neighbor_idx] = True

    # Buffer atoms should not overwrite core atoms
    buffer_mask = buffer_mask & ~core_mask

    # Atoms to keep
    keep_mask = core_mask | buffer_mask
    keep_indices = np.where(keep_mask)[0]

    # If no atoms are found (should not happen since target_atoms are included), return empty
    if len(keep_indices) == 0:
        return Atoms()

    # Create the cluster
    cluster = structure[keep_indices].copy()  # type: ignore[no-untyped-call]

    # Set force weights: 1.0 for core, 0.0 for buffer
    cluster_force_weight = np.zeros(len(cluster))

    # The new indices in the cluster correspond to keep_indices
    # We can map the masks directly
    # Core atoms in the original structure mapped to the cluster
    cluster_core_mask = core_mask[keep_indices]
    cluster_buffer_mask = buffer_mask[keep_indices]

    cluster_force_weight[cluster_core_mask] = 1.0
    cluster_force_weight[cluster_buffer_mask] = 0.0

    cluster.set_array("force_weight", cluster_force_weight)

    # For the cluster, we typically remove periodic boundary conditions to act as a free cluster
    # or place it in a large bounding box. The spec says "stand-alone ASE Atoms object", "vacuum layer added".
    # Let's remove PBC and set a large cell just in case.
    cluster.set_pbc(False)
    # Center the cluster
    cluster.center(vacuum=10.0)

    # Pre-relaxation (to be implemented)
    if config.enable_pre_relaxation and calculator is not None:
        cluster = _pre_relax_buffer(cluster, calculator)

    # Auto-passivation (to be implemented)
    if config.enable_passivation:
        cluster = _passivate_surface(cluster, config.passivation_element)

    return cluster  # type: ignore[no-any-return]


def _pre_relax_buffer(cluster: Atoms, calculator: Calculator) -> Atoms:
    """
    Relaxes the buffer layer using the provided calculator while keeping the core fixed.
    """
    # Create a copy so we don't accidentally modify the input object directly if not wanted
    # But ASE optimization works in-place on the Atoms object.

    # Identify core atoms (force_weight == 1.0)
    force_weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    core_indices = np.where(force_weights == 1.0)[0]

    # Apply constraints
    constraint = FixAtoms(indices=core_indices)  # type: ignore[no-untyped-call]
    cluster.set_constraint(constraint)  # type: ignore[no-untyped-call]

    # Set calculator
    cluster.calc = calculator

    # Run optimization
    opt = LBFGS(cluster, logfile=None)
    # Perform a few steps or until convergence.
    # UAT / SPEC says "Run an LBFGS optimization on the cluster".
    opt.run(fmax=0.05, steps=50)  # type: ignore[no-untyped-call]

    # Remove constraints and calculator after relaxation to leave a clean object
    cluster.set_constraint()  # type: ignore[no-untyped-call]
    cluster.calc = None

    return cluster


def _passivate_surface(cluster: Atoms, element: str) -> Atoms:
    """
    Identifies under-coordinated atoms at the boundary and adds the specified element to passivate.
    """
    if element not in atomic_numbers:
        err_msg = f"Invalid element for passivation: {element}"
        raise ValueError(err_msg)

    passivation_atomic_number = atomic_numbers[element]
    passivation_radius = covalent_radii[passivation_atomic_number]

    force_weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
    buffer_indices = np.where(force_weights == 0.0)[0]

    if len(buffer_indices) == 0:
        return cluster

    # Use neighbor list to find coordination
    # standard bond distance approximation: r_cov(A) + r_cov(B)
    # We will use a multiplier for tolerance
    radii = [covalent_radii[a.number] for a in cluster]

    # Using natural cutoffs from covalent radii + a tolerance (e.g. 0.3 A)
    # to determine bonding
    cutoffs = [r + 0.3 for r in radii]
    i, j, d = neighbor_list("ijd", cluster, cutoffs)  # type: ignore[no-untyped-call]

    new_atoms = Atoms()
    new_force_weights = []

    # Simple heuristic for missing bonds
    # Ideal coordination number is a complex topic. For MgO, bulk coordination is 6.
    # For a robust, generic passivation:
    # If a buffer atom is "surface-like" (has missing neighbors compared to a fully coordinated state),
    # we add passivation atoms along the outward vector.

    # To determine an "outward" vector, we can use the vector from the cluster center of mass
    com = cluster.get_center_of_mass()  # type: ignore[no-untyped-call]

    # We'll apply passivation to buffer atoms that have coordination < expected
    # As a simplified heuristic for "dangling bonds" defined in the spec:
    # We passivate buffer atoms that have "vacuum" around them.
    # Let's add 1 passivation atom to each buffer atom pointing outward.
    # A more sophisticated approach would count exact valences, but the spec says:
    # "calculate a normalized vector pointing outwards from the cluster center and add the element at a standard bond distance."

    for idx in buffer_indices:
        atom = cluster[idx]
        atom_pos = atom.position

        # Vector from center of mass to atom
        outward_vector = atom_pos - com
        norm = np.linalg.norm(outward_vector)
        # Atom is exactly at center, shouldn't happen for buffer, but just in case
        outward_vector = np.array([0.0, 0.0, 1.0]) if norm < 1e-5 else outward_vector / norm

        # Standard bond distance: covalent_radii[atom] + covalent_radii[passivation_element]
        bond_dist = covalent_radii[atom.number] + passivation_radius

        # New position
        new_pos = atom_pos + outward_vector * bond_dist

        # Add to new atoms list
        new_atoms += Atoms(element, positions=[new_pos])
        new_force_weights.append(0.0)

    if len(new_atoms) > 0:
        cluster += new_atoms
        # Update force weights array
        updated_force_weights = np.concatenate([force_weights, np.array(new_force_weights)])
        cluster.set_array("force_weight", updated_force_weights)  # type: ignore[no-untyped-call]

    return cluster
