from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    @model_validator(mode='after')
    def check_cutoff_supercell(self) -> 'PyAceConfig':
        # Future enhancement: Add lattice_constants to StructureConfig or infer from a seed structure.
        return self
