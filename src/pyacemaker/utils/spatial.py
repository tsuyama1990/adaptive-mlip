import logging

import ase
import numpy as np
import numpy.typing as npt

from pyacemaker.domain_models.scenario import SpatialAction, SpatialRegion

logger = logging.getLogger(__name__)

# Action priority for conflict resolution. Higher value = higher priority.
ACTION_PRIORITY: dict[SpatialAction, int] = {
    SpatialAction.ACTION_ACTIVE_LEARNING_ONLY: 1,
    SpatialAction.ACTION_LANGEVIN_THERMOSTAT: 2,
    SpatialAction.ACTION_FREEZE: 3,
}


def apply_spatial_tags(atoms: ase.Atoms, regions: list[SpatialRegion]) -> npt.NDArray[np.int_]:
    """
    Applies spatial tags to an ASE Atoms object based on defined regions.
    Uses purely vectorized numpy operations.

    Args:
        atoms: The ASE Atoms object (initial state).
        regions: List of SpatialRegion models defining the bounding boxes.

    Returns:
        A 1D numpy array of integers representing the tags. 0 means no tag.
        Tags are assigned sequentially starting from 1 for each unique action type.
    """
    from pyacemaker.domain_models.defaults import DEFAULT_TAG_ASSIGNMENT_STRATEGY

    num_atoms = len(atoms)
    tags = np.zeros(num_atoms, dtype=np.int_)
    if num_atoms == 0 or not regions:
        return tags

    cell_lengths = atoms.get_cell().lengths()  # type: ignore[no-untyped-call]
    for i, region in enumerate(regions):
        # Validate region coordinates are within cell dimensions (using generous bounds for logic simplicity)
        if (
            region.x_max < -cell_lengths[0] * 10
            or region.x_min > cell_lengths[0] * 10
            or region.y_max < -cell_lengths[1] * 10
            or region.y_min > cell_lengths[1] * 10
            or region.z_max < -cell_lengths[2] * 10
            or region.z_min > cell_lengths[2] * 10
        ):
            logger.warning(
                f"Region {i + 1} is entirely outside reasonable bounds for cell of size {cell_lengths}."
            )

    # Handle periodic boundaries cleanly by wrapping positions
    # as per architecture "intelligently wrap or clip the selection"
    positions: npt.NDArray[np.float64] = atoms.get_positions(wrap=True)  # type: ignore[no-untyped-call]

    # We will keep track of the priority of the currently assigned tag
    # to handle overlap resolution deterministically.
    current_priorities = np.zeros(num_atoms, dtype=np.int_)

    for i, region in enumerate(regions):
        tag_id = i + 1

        # Calculate bounding mask using logical_and
        # Note: architecture specifies "inclusive on the lower bound, exclusive on the upper bound"
        mask_x = np.logical_and(positions[:, 0] >= region.x_min, positions[:, 0] < region.x_max)
        mask_y = np.logical_and(positions[:, 1] >= region.y_min, positions[:, 1] < region.y_max)
        mask_z = np.logical_and(positions[:, 2] >= region.z_min, positions[:, 2] < region.z_max)

        # Combine all masks
        mask_xyz = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

        num_selected = np.sum(mask_xyz)
        if num_selected == 0:
            logger.warning("SpatialRegion %d (%s) contains zero atoms.", i + 1, region.action)
            continue

        region_priority = ACTION_PRIORITY.get(region.action, 0)

        # Check for conflicts where the new region has a lower or equal priority
        # than the already assigned tag.
        overlap_mask = np.logical_and(mask_xyz, current_priorities > 0)

        # Priority resolution using configurable strategy (defaults to "priority")
        if DEFAULT_TAG_ASSIGNMENT_STRATEGY == "priority":
            for idx in np.where(overlap_mask)[0]:
                existing_priority = current_priorities[idx]
                if region_priority > existing_priority:
                    # New region overrides
                    pass
                elif region_priority < existing_priority:
                    # Existing region is preserved, remove from current mask
                    mask_xyz[idx] = False
                else:
                    # Same priority - unresolvable conflict
                    msg = (
                        f"Unresolvable spatial region conflict at atom index {idx} "
                        f"between region {tags[idx]} and region {tag_id} "
                        f"(both have priority {region_priority})."
                    )
                    raise ValueError(msg)

        # Apply the tag
        tags[mask_xyz] = tag_id
        current_priorities[mask_xyz] = region_priority

    return tags
