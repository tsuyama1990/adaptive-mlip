
from pydantic import BaseModel, ConfigDict, Field

from .dft import DFTConfig
from .distillation import DistillationConfig
from .eon import EONConfig
from .logging import LoggingConfig
from .md import MDConfig
from .scenario import ScenarioConfig
from .structure import StructureConfig
from .training import TrainingConfig
from .validation import ValidationConfig
from .workflow import WorkflowConfig


import os

class PyAceConfig(BaseModel):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # Determine extra parameter based on env var or config hook in future
        pass

    model_config = ConfigDict(extra=os.environ.get("PYACEMAKER_CONFIG_EXTRA", "forbid")) # type: ignore[misc]

    project_name: str = Field(..., min_length=1, description="Name of the project")
    structure: StructureConfig
    dft: DFTConfig
    training: TrainingConfig
    md: MDConfig
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig, description="Validation configuration"
    )
    workflow: WorkflowConfig
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    eon: EONConfig | None = Field(None, description="EON configuration")
    scenario: ScenarioConfig | None = Field(None, description="Scenario configuration")
    distillation: DistillationConfig | None = Field(None, description="Distillation phase configuration")
