from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScenarioConfig(BaseModel):
    """Configuration for specific scenarios."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Name of the scenario to run")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Scenario-specific parameters"
    )
    enabled: bool = Field(False, description="Whether to run this scenario")
