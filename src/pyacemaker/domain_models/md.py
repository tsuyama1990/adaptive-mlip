from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: PositiveFloat = Field(..., description="Simulation temperature in Kelvin")
    pressure: float = Field(..., description="Simulation pressure in Bar")
    timestep: PositiveFloat = Field(..., description="Timestep in ps")
    n_steps: int = Field(..., gt=0, description="Number of MD steps")

    # Spec Section 3.4 (OTF)
    uncertainty_threshold: float = Field(
        5.0, gt=0.0, description="Gamma threshold for halting simulation"
    )
    check_interval: int = Field(
        10, gt=0, description="Step interval for uncertainty check"
    )
