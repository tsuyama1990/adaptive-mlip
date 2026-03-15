from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpatialAction(StrEnum):
    ACTION_FREEZE = "ACTION_FREEZE"
    ACTION_LANGEVIN_THERMOSTAT = "ACTION_LANGEVIN_THERMOSTAT"
    ACTION_ACTIVE_LEARNING_ONLY = "ACTION_ACTIVE_LEARNING_ONLY"


class SpatialRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)

    x_min: float = Field(..., description="Minimum x coordinate in Angstroms")
    x_max: float = Field(..., description="Maximum x coordinate in Angstroms")
    y_min: float = Field(..., description="Minimum y coordinate in Angstroms")
    y_max: float = Field(..., description="Maximum y coordinate in Angstroms")
    z_min: float = Field(..., description="Minimum z coordinate in Angstroms")
    z_max: float = Field(..., description="Maximum z coordinate in Angstroms")
    action: SpatialAction = Field(..., description="Physical action to apply to the region")

    @field_validator("action", mode="before")
    @classmethod
    def convert_action(cls, v: str | SpatialAction) -> SpatialAction:
        if isinstance(v, str):
            return SpatialAction(v)
        return v

    @model_validator(mode="after")
    def validate_boundaries(self) -> "SpatialRegion":
        if self.x_min > self.x_max:
            msg = f"Invalid x-axis boundary: x_min ({self.x_min}) > x_max ({self.x_max})"
            raise ValueError(msg)
        if self.y_min > self.y_max:
            msg = f"Invalid y-axis boundary: y_min ({self.y_min}) > y_max ({self.y_max})"
            raise ValueError(msg)
        if self.z_min > self.z_max:
            msg = f"Invalid z-axis boundary: z_min ({self.z_min}) > z_max ({self.z_max})"
            raise ValueError(msg)
        return self
