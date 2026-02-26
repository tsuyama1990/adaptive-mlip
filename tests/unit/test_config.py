import pytest
from pydantic import ValidationError
from pyacemaker.domain_models.structure import StructureConfig, ExplorationPolicy

def test_config_r_cut_positive():
    """Test that r_cut must be positive."""
    with pytest.raises(ValidationError):
        StructureConfig(
            elements=["Fe"],
            supercell_size=[2, 2, 2],
            num_structures=1,
            r_cut=0.0
        )
    with pytest.raises(ValidationError):
        StructureConfig(
            elements=["Fe"],
            supercell_size=[2, 2, 2],
            num_structures=1,
            r_cut=-1.0
        )

def test_config_elements_non_empty():
    """Test that elements list cannot be empty."""
    with pytest.raises(ValidationError):
        StructureConfig(
            elements=[],
            supercell_size=[2, 2, 2],
            num_structures=1
        )

def test_config_supercell_positive():
    """Test that supercell dimensions must be positive."""
    with pytest.raises(ValidationError):
        StructureConfig(
            elements=["Fe"],
            supercell_size=[0, 2, 2],
            num_structures=1
        )
    with pytest.raises(ValidationError):
        StructureConfig(
            elements=["Fe"],
            supercell_size=[-1, 2, 2],
            num_structures=1
        )

def test_config_defaults():
    """Test default values."""
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        num_structures=10
    )
    assert config.r_cut == 2.0 # Default value from StructureConfig definition
    assert config.active_policies == [ExplorationPolicy.COLD_START]

def test_config_duplicates():
    """Test duplicate elements are rejected."""
    with pytest.raises(ValidationError):
        StructureConfig(
            elements=["Fe", "Fe"],
            supercell_size=[2, 2, 2],
            num_structures=1
        )
