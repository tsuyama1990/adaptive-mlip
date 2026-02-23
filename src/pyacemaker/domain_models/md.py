from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: PositiveFloat = Field(..., description="Simulation temperature in Kelvin")
    pressure: float = Field(..., ge=0.0, description="Simulation pressure in Bar")
    timestep: PositiveFloat = Field(..., description="Timestep in ps")
    n_steps: int = Field(..., gt=0, description="Number of MD steps")

    @field_validator("n_steps")
    @classmethod
    def validate_simulation_time(cls, v: int, _info: Any) -> int:
        """Ensure total simulation time is reasonable (e.g. > 0)."""
        # We need access to timestep, but field validation happens in order or we use model_validator.
        # Simple check for n_steps > 0 is already done by gt=0.
        # If we want to check timestep * n_steps, we need model_validator.
        return v

    # Spec Section 3.4 (OTF)
    uncertainty_threshold: float = Field(
        5.0, gt=0.0, description="Gamma threshold for halting simulation"
    )
    check_interval: int = Field(
        10, gt=0, description="Step interval for uncertainty check"
    )
