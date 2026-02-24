import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.scenario import ScenarioConfig


def test_scenario_config_valid() -> None:
    """Test creating a valid scenario configuration."""
    config = ScenarioConfig(
        name="fept_mgo",
        output_dir="test_run",
        deposition_count=50,
        mock=True,
    )
    assert config.name == "fept_mgo"
    assert config.output_dir == "test_run"
    assert config.deposition_count == 50
    assert config.mock is True


def test_scenario_config_defaults() -> None:
    """Test defaults for scenario configuration."""
    config = ScenarioConfig(name="base")
    assert config.deposition_count == 100
    assert config.slab_size == (2, 2, 1)
    assert config.mock is False


def test_scenario_config_invalid_name() -> None:
    """Test invalid scenario name raises ValidationError."""
    with pytest.raises(ValidationError):
        ScenarioConfig(name="invalid_scenario")


def test_scenario_config_invalid_count() -> None:
    """Test invalid deposition count raises ValidationError."""
    with pytest.raises(ValidationError):
        ScenarioConfig(name="fept_mgo", deposition_count=-1)


def test_scenario_config_extra_field() -> None:
    """Test extra fields are forbidden."""
    with pytest.raises(ValidationError):
        ScenarioConfig(name="fept_mgo", unknown="extra")
