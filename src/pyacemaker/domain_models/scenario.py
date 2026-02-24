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
    rng_seed: int = Field(12345, description="Random seed for procedural generation")

    # Scenario specific parameters (for FePt/MgO)
    deposition_count: int = Field(100, ge=1, description="Number of atoms to deposit (FePt)")

    # Surface generation parameters
    slab_size: tuple[int, int, int] = Field((2, 2, 1), description="Slab size (u, v, w)")
    vacuum_size: float = Field(10.0, gt=0, description="Vacuum size in Angstroms")
    deposition_height_offset: float = Field(3.0, description="Height offset for deposition (Angstroms)")

    # Visualization
    visualization_rotation: str = Field("10x,10y,0z", description="Rotation string for visualization")

    # Enable mock mode for faster testing
    mock: bool = Field(False, description="Run in mock mode (skips heavy calculations)")
