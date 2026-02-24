from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ScenarioConfig(BaseModel):
    """
    Configuration for selecting and running specific scenarios.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["fept_mgo", "base"] = Field(..., description="Name of the scenario to run")

    # Generic scenario parameters
    output_dir: str | None = Field(None, description="Output directory for the scenario")

    # Scenario specific parameters (for FePt/MgO)
    # Could be refactored into subclasses if needed, but for now we keep it simple
    deposition_count: int = Field(100, ge=1, description="Number of atoms to deposit (FePt)")

    # Surface generation parameters
    slab_size: tuple[int, int, int] = Field((2, 2, 1), description="Slab size (u, v, w)")

    # Enable mock mode for faster testing
    mock: bool = Field(False, description="Run in mock mode (skips heavy calculations)")
