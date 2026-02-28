from pydantic import BaseModel, ConfigDict, Field

from .dft import DFTConfig
from .eon import EONConfig
from .logging import LoggingConfig
from .md import MDConfig
from .scenario import ScenarioConfig
from .structure import StructureConfig
from .training import TrainingConfig
from .validation import ValidationConfig
from .workflow import WorkflowConfig


class DistillationConfig(BaseModel):
    """Configures Phase 1 (Zero-Shot Distillation)."""
    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=True)
    mace_model_path: str = Field(default="mace-mp-0-medium")
    uncertainty_threshold: float = Field(default=0.05, description="Threshold below which MACE is confident")
    sampling_structures_per_system: int = Field(default=1000)


class ActiveLearningThresholds(BaseModel):
    """Two-tier threshold system inspired by FLARE."""
    model_config = ConfigDict(extra="forbid")

    threshold_call_dft: float = Field(default=0.05, description="System-wide max uncertainty to halt MD")
    threshold_add_train: float = Field(default=0.02, description="Per-atom uncertainty to include in learning set")
    smooth_steps: int = Field(default=3, description="Consecutive steps threshold must be exceeded")


class CutoutConfig(BaseModel):
    """Configures Phase 3 (Intelligent Cutout)."""
    model_config = ConfigDict(extra="forbid")

    core_radius: float = Field(default=4.0, ge=0.0, description="Radius for core atoms (weight=1.0)")
    buffer_radius: float = Field(default=3.0, ge=0.0, description="Radius for buffer atoms (weight=0.0)")
    enable_pre_relaxation: bool = Field(default=True)
    enable_passivation: bool = Field(default=True)
    passivation_element: str = Field(default="H")


class LoopStrategyConfig(BaseModel):
    """Configures overall active learning loop strategy."""
    model_config = ConfigDict(extra="forbid")

    use_tiered_oracle: bool = Field(default=True)
    incremental_update: bool = Field(default=True)
    replay_buffer_size: int = Field(default=500, gt=0)
    baseline_potential_type: str = Field(default="LJ")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)


class PyAceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    distillation: DistillationConfig | None = Field(None, description="Phase 1 configuration")
    cutout: CutoutConfig | None = Field(None, description="Phase 3 cutout configuration")
    loop_strategy: LoopStrategyConfig | None = Field(None, description="Overall active learning loop strategy")
