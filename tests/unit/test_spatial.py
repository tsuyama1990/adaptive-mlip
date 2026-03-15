import numpy as np
import pytest
from ase.build import bulk
from pyacemaker.domain_models.gui_schema import PhysicalAction, SpatialRegion
from pyacemaker.utils.spatial import apply_spatial_tags

def test_spatial_tagging_basic_cube():
    # 10x10x10 SC lattice of Cu -> 1000 atoms
    atoms = bulk("Cu", "sc", a=1.0).repeat((10, 10, 10))

    # Bounding box z_min=0.0, z_max=5.0
    regions = [
        SpatialRegion(
            x_min=-1.0, x_max=11.0,
            y_min=-1.0, y_max=11.0,
            z_min=0.0, z_max=5.0,
            action=PhysicalAction.ACTION_FREEZE
        )
    ]

    tags = apply_spatial_tags(atoms, regions)

    # 5x10x10 atoms should be tagged
    assert np.sum(tags == 1) == 500
    assert np.sum(tags == 0) == 500

def test_spatial_tagging_inclusive_exclusive_edges():
    atoms = bulk("Cu", "sc", a=1.0).repeat((1, 1, 3))
    # Positions are roughly z=0, 1, 2
    # z_min=1.0 (inclusive), z_max=2.0 (exclusive) -> only atom at z=1.0 should be included

    regions = [
        SpatialRegion(
            x_min=-1.0, x_max=2.0,
            y_min=-1.0, y_max=2.0,
            z_min=1.0, z_max=2.0,
            action=PhysicalAction.ACTION_FREEZE
        )
    ]

    tags = apply_spatial_tags(atoms, regions)
    assert np.sum(tags == 1) == 1
    assert tags[1] == 1 # Second atom at z=1.0 is tagged

def test_spatial_tagging_conflict_resolution():
    atoms = bulk("Cu", "sc", a=1.0).repeat((1, 1, 10))

    # Region 1 (FREEZE) z: 0-5
    # Region 2 (THERMOSTAT) z: 3-8
    # Conflict is 3-5
    # FREEZE has higher priority

    regions = [
        SpatialRegion(
            x_min=-1.0, x_max=2.0, y_min=-1.0, y_max=2.0, z_min=0.0, z_max=5.0,
            action=PhysicalAction.ACTION_FREEZE
        ),
        SpatialRegion(
            x_min=-1.0, x_max=2.0, y_min=-1.0, y_max=2.0, z_min=3.0, z_max=8.0,
            action=PhysicalAction.ACTION_LANGEVIN_THERMOSTAT
        )
    ]

    tags = apply_spatial_tags(atoms, regions)

    # Atoms 0,1,2 should be 1
    # Atoms 3,4 should be 1 (FREEZE overrides)
    # Atoms 5,6,7 should be 2
    # Atoms 8,9 should be 0
    assert list(tags) == [1, 1, 1, 1, 1, 2, 2, 2, 0, 0]

def test_spatial_tagging_empty_region():
    atoms = bulk("Cu", "sc", a=1.0).repeat((1, 1, 1))

    regions = [
        SpatialRegion(
            x_min=100.0, x_max=110.0, y_min=-1.0, y_max=2.0, z_min=-1.0, z_max=2.0,
            action=PhysicalAction.ACTION_FREEZE
        )
    ]

    tags = apply_spatial_tags(atoms, regions)
    assert np.sum(tags == 1) == 0

def test_schema_inverted_coordinates():
    with pytest.raises(ValueError, match="Inverted Z coordinates"):
        SpatialRegion(
            x_min=-1.0, x_max=11.0,
            y_min=-1.0, y_max=11.0,
            z_min=5.0, z_max=0.0,
            action=PhysicalAction.ACTION_FREEZE
        )
