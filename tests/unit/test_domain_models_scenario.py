import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.scenario import ScenarioConfig


def test_scenario_config_valid() -> None:
    config = ScenarioConfig(
        name="fept_mgo",
        parameters={"temperature": 500.0, "steps": 100},
        enabled=True,
    )
    assert config.name == "fept_mgo"
    assert config.parameters == {"temperature": 500.0, "steps": 100}
    assert config.enabled


def test_scenario_config_defaults() -> None:
    config = ScenarioConfig(name="test")
    assert not config.enabled
    assert config.parameters == {}


def test_scenario_config_invalid_name() -> None:
    with pytest.raises(ValidationError):
        ScenarioConfig()  # Missing name
