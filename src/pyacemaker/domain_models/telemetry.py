from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SimulationState(StrEnum):
    RUNNING_MD = "RUNNING_MD"
    EVALUATING_UNCERTAINTY = "EVALUATING_UNCERTAINTY"
    EXTRACTING_CUTOUT = "EXTRACTING_CUTOUT"
    RUNNING_DFT = "RUNNING_DFT"
    TRAINING_MACE = "TRAINING_MACE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class SystemTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    workflow_id: str = Field("default_workflow", description="ID of the workflow")
    atomic_numbers: list[int] = Field(..., description="Array of atomic numbers")
    total_atoms: int = Field(..., description="Total number of atoms in the system")
    cell_dimensions: list[float] | None = Field(None, description="Flattened 3x3 cell matrix [xx, xy, xz, yx, yy, yz, zx, zy, zz]")

class TelemetryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    workflow_id: str = Field("default_workflow", description="ID of the workflow")
    step_number: int = Field(..., ge=0, description="Current simulation timestep")
    current_state: SimulationState = Field(..., description="Current state of the orchestrator")
    positions: list[float] = Field(..., description="Flattened 1D array of Cartesian coordinates [x1, y1, z1, x2, y2, z2...]")
    forces: list[float] | None = Field(None, description="Flattened 1D array of force vectors")
    variances: list[float] | None = Field(None, description="Array of atomic uncertainty metrics")

class StateChangePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    workflow_id: str = Field("default_workflow", description="ID of the workflow")
    type: str = Field(default="state_change", description="Message type")
    new_state: SimulationState = Field(..., description="The new simulation state")
