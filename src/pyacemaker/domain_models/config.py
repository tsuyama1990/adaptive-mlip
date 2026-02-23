from pydantic import BaseModel, ConfigDict, Field

from .dft import DFTConfig
from .logging import LoggingConfig
from .md import MDConfig
from .structure import StructureConfig
from .training import TrainingConfig
from .workflow import WorkflowConfig


class PyAceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(..., min_length=1, description="Name of the project")
    structure: StructureConfig
    dft: DFTConfig
    training: TrainingConfig
    md: MDConfig
    workflow: WorkflowConfig
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
