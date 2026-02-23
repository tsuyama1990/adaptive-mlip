from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: PositiveFloat = Field(..., description="Simulation temperature in Kelvin")
    pressure: float = Field(default=0.0, description="Simulation pressure in Bar")
    timestep: PositiveFloat = Field(default=0.001, description="Timestep in ps")
    n_steps: int = Field(default=1000, gt=0, description="Number of MD steps")
