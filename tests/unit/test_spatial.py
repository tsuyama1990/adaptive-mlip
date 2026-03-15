import numpy as np
import pytest
from ase.build import bulk

from pyacemaker.domain_models.gui_schema import SpatialAction, SpatialRegion
from pyacemaker.utils.spatial import apply_spatial_tags, get_tag_for_action


def test_get_tag_for_action() -> None:
    assert get_tag_for_action(SpatialAction.ACTION_FREEZE) == 1
    assert get_tag_for_action(SpatialAction.ACTION_LANGEVIN_THERMOSTAT) == 2
    assert get_tag_for_action(SpatialAction.ACTION_ACTIVE_LEARNING_ONLY) == 3

    with pytest.raises(ValueError, match="Unknown spatial action"):
        get_tag_for_action("INVALID_ACTION")  # type: ignore[arg-type]


def test_spatial_region_validation() -> None:
    # Valid region
    region = SpatialRegion(
        x_min=0.0,
        x_max=5.0,
        y_min=-1.0,
        y_max=2.0,
        z_min=0.0,
        z_max=10.0,
        action=SpatialAction.ACTION_FREEZE,
    )
    assert region.x_max == 5.0

    # Invalid region (min > max)
    with pytest.raises(ValueError, match="Invalid z-axis boundary: z_min \\(10.0\\) > z_max \\(5.0\\)"):
        SpatialRegion(
            x_min=0.0,
            x_max=5.0,
            y_min=0.0,
            y_max=5.0,
            z_min=10.0,
            z_max=5.0,
            action=SpatialAction.ACTION_FREEZE,
        )


def test_apply_spatial_tags_empty() -> None:
    atoms = bulk("Cu", "fcc", a=3.6)
    tags = apply_spatial_tags(atoms, [])
    assert len(tags) == len(atoms)
    assert np.all(tags == 0)


def test_apply_spatial_tags_basic() -> None:
    # 2x2x2 supercell = 8 atoms
    atoms = bulk("Cu", "sc", a=2.0).repeat((2, 2, 2))  # type: ignore[no-untyped-call]
    # Positions are: (0,0,0), (2,0,0), (0,2,0), (2,2,0), (0,0,2), (2,0,2), (0,2,2), (2,2,2)
    # Plus periodic boundaries handled by repeat.
    # We want to select z=0 layer
    region = SpatialRegion(
        x_min=-1.0,
        x_max=5.0,
        y_min=-1.0,
        y_max=5.0,
        z_min=-0.1,
        z_max=0.1,  # Only z=0 atoms
        action=SpatialAction.ACTION_FREEZE,
    )

    tags = apply_spatial_tags(atoms, [region])

    # 4 atoms at z=0, 4 atoms at z=2
    assert len(tags) == 8
    assert np.sum(tags == 1) == 4
    assert np.sum(tags == 0) == 4

    positions = atoms.get_positions()  # type: ignore[no-untyped-call]
    for i in range(8):
        if positions[i, 2] == 0.0:
            assert tags[i] == 1
        else:
            assert tags[i] == 0


def test_apply_spatial_tags_overlaps_deterministic_resolution() -> None:
    atoms = bulk("Cu", "sc", a=2.0).repeat((1, 1, 3))  # type: ignore[no-untyped-call]
    # Positions: z=0, z=2, z=4

    region_thermostat = SpatialRegion(
        x_min=-1.0, x_max=3.0, y_min=-1.0, y_max=3.0, z_min=-1.0, z_max=5.0,
        action=SpatialAction.ACTION_LANGEVIN_THERMOSTAT
    )

    region_freeze = SpatialRegion(
        x_min=-1.0, x_max=3.0, y_min=-1.0, y_max=3.0, z_min=-1.0, z_max=2.5,
        action=SpatialAction.ACTION_FREEZE
    )

    # thermostat applies to all (z=0, 2, 4)
    # freeze applies to z=0, z=2
    # FREEZE has priority, so z=0 and z=2 get 1, z=4 gets 2.

    tags = apply_spatial_tags(atoms, [region_thermostat, region_freeze])

    assert len(tags) == 3
    positions = atoms.get_positions()  # type: ignore[no-untyped-call]

    for i in range(3):
        if positions[i, 2] <= 2.0:
            assert tags[i] == 1
        else:
            assert tags[i] == 2


def test_apply_spatial_tags_conflict_error() -> None:
    atoms = bulk("Cu", "sc", a=2.0)

    region1 = SpatialRegion(
        x_min=-1.0, x_max=3.0, y_min=-1.0, y_max=3.0, z_min=-1.0, z_max=3.0,
        action=SpatialAction.ACTION_LANGEVIN_THERMOSTAT
    )

    region2 = SpatialRegion(
        x_min=-1.0, x_max=3.0, y_min=-1.0, y_max=3.0, z_min=-1.0, z_max=3.0,
        action=SpatialAction.ACTION_ACTIVE_LEARNING_ONLY
    )

    with pytest.raises(ValueError, match="Deterministic conflict resolution failed"):
        apply_spatial_tags(atoms, [region1, region2])
