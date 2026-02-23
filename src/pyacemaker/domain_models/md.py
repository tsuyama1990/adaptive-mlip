
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: PositiveFloat = Field(..., description="Simulation temperature in Kelvin")
    pressure: float = Field(..., ge=0.0, description="Simulation pressure in Bar")
    timestep: PositiveFloat = Field(..., description="Timestep in ps")
    n_steps: int = Field(..., gt=0, description="Number of MD steps")

    # Mocking Parameters (Audit Requirement)
    base_energy: float = Field(
        -100.0, description="Baseline energy for mock simulation"
    )
    default_forces: list[list[float]] = Field(
        default=[[0.0, 0.0, 0.0]], description="Default forces for mock simulation"
    )

    # Spec Section 3.4 (Hybrid Potential & OTF)
    hybrid_potential: bool = Field(
        False, description="Use hybrid potential (ACE + LJ/ZBL)"
    )
    hybrid_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for hybrid potential baseline"
    )

    # Spec Section 3.4 (OTF)
    uncertainty_threshold: float = Field(
        5.0, gt=0.0, description="Gamma threshold for halting simulation"
    )
    check_interval: int = Field(
        10, gt=0, description="Step interval for uncertainty check"
    )
