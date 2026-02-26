import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.scenario import ScenarioConfig


def test_scenario_config_valid():
    config = ScenarioConfig(
        name="fept_mgo",
        parameters={"fe_pt_ratio": 0.5, "steps": 100},
        enabled=True,
    )
    assert config.name == "fept_mgo"
    assert config.parameters["fe_pt_ratio"] == 0.5
    assert config.enabled is True


def test_scenario_config_extra_forbid():
    with pytest.raises(ValidationError):
        ScenarioConfig(name="test", extra_field="forbidden")
