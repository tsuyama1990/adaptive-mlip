from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class PhysicalAction(StrEnum):
    ACTION_FREEZE = "ACTION_FREEZE"
    ACTION_LANGEVIN_THERMOSTAT = "ACTION_LANGEVIN_THERMOSTAT"
    ACTION_ACTIVE_LEARNING_ONLY = "ACTION_ACTIVE_LEARNING_ONLY"

class SpatialRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)

    x_min: float = Field(..., description="Minimum X coordinate")
    x_max: float = Field(..., description="Maximum X coordinate")
    y_min: float = Field(..., description="Minimum Y coordinate")
    y_max: float = Field(..., description="Maximum Y coordinate")
    z_min: float = Field(..., description="Minimum Z coordinate")
    z_max: float = Field(..., description="Maximum Z coordinate")
    action: PhysicalAction = Field(..., description="Physical action to apply to the region")

    @model_validator(mode="after")
    def validate_bounds(self) -> "SpatialRegion":
        if self.x_min > self.x_max:
            msg = f"Inverted X coordinates: x_min ({self.x_min}) > x_max ({self.x_max})"
            raise ValueError(msg)
        if self.y_min > self.y_max:
            msg = f"Inverted Y coordinates: y_min ({self.y_min}) > y_max ({self.y_max})"
            raise ValueError(msg)
        if self.z_min > self.z_max:
            msg = f"Inverted Z coordinates: z_min ({self.z_min}) > z_max ({self.z_max})"
            raise ValueError(msg)
        return self
