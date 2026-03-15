import ase
import numpy as np
from numpy.typing import NDArray

from pyacemaker.domain_models.gui_schema import PhysicalAction, SpatialRegion

def apply_spatial_tags(atoms: ase.Atoms, regions: list[SpatialRegion]) -> NDArray[np.int_]:
    """
    Applies spatial tags to an ASE Atoms object based on mathematically defined bounding regions.
    Uses precise boolean masking on coordinates. The lower bounds are inclusive and upper bounds are exclusive.
    Returns a 1D NumPy array of integer tags.
    """
    tags = np.zeros(len(atoms), dtype=np.int_)
    if not regions:
        return tags

    positions: NDArray[np.float64] = (
        atoms.get_scaled_positions() @ atoms.get_cell() if any(atoms.pbc) else atoms.get_positions()
    )

    priority_map = {
        PhysicalAction.ACTION_ACTIVE_LEARNING_ONLY: 1,
        PhysicalAction.ACTION_LANGEVIN_THERMOSTAT: 2,
        PhysicalAction.ACTION_FREEZE: 3,
    }

    current_priorities = np.zeros(len(atoms), dtype=np.int_)

    for i, region in enumerate(regions):
        tag_value = i + 1

        mask_x = np.logical_and(positions[:, 0] >= region.x_min, positions[:, 0] < region.x_max)
        mask_y = np.logical_and(positions[:, 1] >= region.y_min, positions[:, 1] < region.y_max)
        mask_z = np.logical_and(positions[:, 2] >= region.z_min, positions[:, 2] < region.z_max)

        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        if not np.any(mask):
            import logging

            logging.getLogger(__name__).warning(
                f"Spatial region {i} ({region.action}) contains zero atoms."
            )
            continue

        region_priority = priority_map[region.action]

        equal_priority_mask = np.logical_and(
            np.logical_and(mask, current_priorities > 0), current_priorities == region_priority
        )
        if np.any(equal_priority_mask):
            msg = f"Unresolvable conflict: Overlapping regions with equal priority ({region.action})."
            raise ValueError(msg)

        update_mask = np.logical_and(mask, current_priorities < region_priority)

        tags[update_mask] = tag_value
        current_priorities[update_mask] = region_priority

    return tags
