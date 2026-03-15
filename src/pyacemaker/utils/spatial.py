import logging
from typing import cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from pyacemaker.domain_models.gui_schema import SpatialAction, SpatialRegion

logger = logging.getLogger(__name__)


def get_tag_for_action(action: SpatialAction) -> int:
    """Maps SpatialAction to specific integer tags."""
    if action == SpatialAction.ACTION_FREEZE:
        return 1
    if action == SpatialAction.ACTION_LANGEVIN_THERMOSTAT:
        return 2
    if action == SpatialAction.ACTION_ACTIVE_LEARNING_ONLY:
        return 3
    msg = f"Unknown spatial action: {action}"
    raise ValueError(msg)


def apply_spatial_tags(atoms: Atoms, regions: list[SpatialRegion]) -> NDArray[np.int_]:
    """
    Applies spatial bounding boxes to an ASE Atoms object, returning an array of integer tags.

    The boundary logic is mathematically inclusive on the lower bound and exclusive
    on the upper bound: min <= position < max. This ensures deterministic handling
    if an atom falls exactly on the boundary plane.

    Overlapping regions are resolved using a deterministic priority hierarchy.
    ACTION_FREEZE (tag 1) has the highest priority and will overwrite others.
    If regions have the same action, they simply union. If they have conflicting
    actions with the same implicit priority level, a ValueError is raised to fail fast.
    """
    if not regions:
        return np.zeros(len(atoms), dtype=np.int_)

    # Copy the positions. Depending on whether atoms have been wrapped or scaled,
    # the user is specifying Cartesian boundaries on the provided state.
    positions = cast(NDArray[np.float64], atoms.get_positions())  # type: ignore[no-untyped-call]
    n_atoms = len(atoms)

    # Output tags array, initialized to 0 (default/no tag)
    final_tags = np.zeros(n_atoms, dtype=np.int_)

    # Process regions and gather their boolean masks
    for region in regions:
        x_min, x_max = region.x_min, region.x_max
        y_min, y_max = region.y_min, region.y_max
        z_min, z_max = region.z_min, region.z_max

        mask_x = np.logical_and(positions[:, 0] >= x_min, positions[:, 0] < x_max)
        mask_y = np.logical_and(positions[:, 1] >= y_min, positions[:, 1] < y_max)
        mask_z = np.logical_and(positions[:, 2] >= z_min, positions[:, 2] < z_max)

        # Atoms inside this 3D box
        region_mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

        # Count selected atoms
        num_selected = np.sum(region_mask)
        if num_selected == 0:
            logger.warning(
                f"SpatialRegion with action {region.action.value} "
                f"from ({x_min}, {y_min}, {z_min}) to ({x_max}, {y_max}, {z_max}) "
                "contains zero atoms."
            )
            continue

        target_tag = get_tag_for_action(region.action)

        # Handle overlaps deterministically
        for i in range(n_atoms):
            if region_mask[i]:
                current_tag = final_tags[i]
                if current_tag == 0:
                    final_tags[i] = target_tag
                elif current_tag == target_tag:
                    # Union of same actions
                    pass
                # Conflict resolution
                # ACTION_FREEZE (1) overrides others
                elif current_tag == 1:
                    # Existing tag is Freeze, it wins. Do nothing.
                    pass
                elif target_tag == 1:
                    # New tag is Freeze, it overrides the existing.
                    final_tags[i] = target_tag
                else:
                    # Conflict between equal priority actions
                    msg = (
                        f"Deterministic conflict resolution failed: Atom {i} "
                        f"is subject to conflicting actions mapping to tags {current_tag} and {target_tag}."
                    )
                    raise ValueError(msg)

    return final_tags
