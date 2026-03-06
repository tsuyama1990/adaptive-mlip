from typing import Any

from pydantic import BaseModel, ConfigDict, Field


from enum import StrEnum

class ScenarioName(StrEnum):
    FEPT_MGO = "fept_mgo"
    # Other scenarios can be added here

class ScenarioConfig(BaseModel):
    """Configuration for specific scenarios."""

    model_config = ConfigDict(extra="forbid")

    name: str | ScenarioName = Field(..., description="Name of the scenario to run")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Scenario-specific parameters"
    )
    enabled: bool = Field(False, description="Whether to run this scenario")
