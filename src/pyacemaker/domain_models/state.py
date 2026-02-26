from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StepState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    message: str = Field(default="")


class LoopState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(default=0)
    max_iterations: int = Field(default=0)
    converged: bool = Field(default=False)


class GlobalState(BaseModel):
    """
    Tracks the global state of the workflow for persistence and resumption.
    """
    model_config = ConfigDict(extra="forbid")

    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    current_step: int = Field(default=0)
    steps: dict[str, StepState] = Field(default_factory=dict)
    loop: LoopState | None = Field(default=None)
