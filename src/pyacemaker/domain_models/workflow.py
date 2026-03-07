from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from pyacemaker.domain_models.defaults import (
    DEFAULT_ACTIVE_LEARNING_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DATA_DIR,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OTF_LOCAL_N_CANDIDATES,
    DEFAULT_OTF_LOCAL_N_SELECT,
    DEFAULT_OTF_MAX_RETRIES,
    DEFAULT_POTENTIALS_DIR,
    DEFAULT_STATE_FILE,
)


class ActiveLearningThresholds(BaseModel):
    """Two-Tier Thresholds inspired by FLARE"""
    model_config = ConfigDict(extra="forbid")

    threshold_call_dft: float = Field(0.05, description="Criteria to halt MD and call DFT")
    threshold_add_train: float = Field(0.02, description="Criteria to select atoms for training set")
    smooth_steps: int = Field(3, description="Consecutive steps threshold exceedance required to eliminate thermal noise")

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ActiveLearningThresholds":
        if self.threshold_add_train > self.threshold_call_dft:
            raise ValueError('threshold_add_train must be <= threshold_call_dft')
        return self

class CutoutConfig(BaseModel):
    """Phase 3: Intelligent Cutout Settings"""
    model_config = ConfigDict(extra="forbid")

    core_radius: float = Field(4.0, description="Radius for Force Weight 1.0")
    buffer_radius: float = Field(3.0, description="Thickness of additional relaxation buffer layer")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

    @model_validator(mode="after")
    def validate_radii(self) -> "CutoutConfig":
        if self.buffer_radius > self.core_radius:
            raise ValueError('buffer_radius must be <= core_radius')
        return self

class DistillationConfig(BaseModel):
    """Phase 1: Zero-Shot Distillation Settings"""
    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="Safe threshold for MACE")
    sampling_structures_per_system: int = Field(1000, ge=100, le=10000, description="Safe threshold for MACE")

class LoopStrategyConfig(BaseModel):
    """Active Learning Loop Strategy Settings"""
    model_config = ConfigDict(extra="forbid")

    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, ge=10, le=10000, description="Number of past data points to retain to prevent catastrophic forgetting")
    baseline_potential_type: str = Field("LJ", description="Physical baseline potential (e.g., LJ)")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)

class OTFConfig(BaseModel):
    """Configuration for On-The-Fly (OTF) Active Learning loop."""

    model_config = ConfigDict(extra="forbid")

    fix_halt: bool = Field(False, description="Enable OTF halting based on uncertainty")
    check_interval: int = Field(
        10, gt=0, le=1000, description="Step interval for uncertainty check"
    )

    local_n_candidates: PositiveInt = Field(
        default=DEFAULT_OTF_LOCAL_N_CANDIDATES,
        description="Number of local candidates to generate around halt structure.",
    )
    local_n_select: PositiveInt = Field(
        default=DEFAULT_OTF_LOCAL_N_SELECT,
        description="Number of candidates to select via active set optimization.",
    )
    max_retries: PositiveInt = Field(
        default=DEFAULT_OTF_MAX_RETRIES,
        description="Maximum number of retraining attempts per iteration.",
    )


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: PositiveInt = Field(..., description="Maximum number of active learning cycles")
    convergence_energy: float = Field(
        default=0.001, ge=1e-6, le=0.1, description="Energy convergence criteria in eV/atom"
    )
    convergence_force: float = Field(
        default=0.01, ge=1e-4, le=1.0, description="Force convergence criteria in eV/Angstrom"
    )
    state_file_path: str = Field(
        default=DEFAULT_STATE_FILE, description="Path to the state checkpoint file"
    )

    # New fields to avoid magic numbers
    batch_size: PositiveInt = Field(
        default=DEFAULT_BATCH_SIZE, description="Number of structures to process in a batch"
    )
    n_candidates: PositiveInt = Field(
        default=DEFAULT_N_CANDIDATES,
        description="Number of candidate structures to generate per iteration",
    )
    checkpoint_interval: PositiveInt = Field(
        default=DEFAULT_CHECKPOINT_INTERVAL, gt=0, description="Save state every N iterations"
    )
    data_dir: str = Field(
        default=DEFAULT_DATA_DIR, description="Directory to store training data and artifacts"
    )
    active_learning_dir: str = Field(
        default=DEFAULT_ACTIVE_LEARNING_DIR, description="Directory for active learning iterations"
    )
    potentials_dir: str = Field(
        default=DEFAULT_POTENTIALS_DIR, description="Directory for storing trained potentials"
    )

    otf: OTFConfig = Field(default_factory=OTFConfig, description="Configuration for OTF loop.")
    loop_strategy: LoopStrategyConfig = Field(default_factory=LoopStrategyConfig, description="Configuration for loop strategy.")
    distillation: DistillationConfig = Field(default_factory=DistillationConfig, description="Configuration for distillation.")
    cutout: CutoutConfig = Field(default_factory=CutoutConfig, description="Configuration for cutout.")

    @model_validator(mode="after")
    def validate_checkpoint_interval(self) -> "WorkflowConfig":
        """Ensures checkpoint_interval is logically sound."""
        if self.checkpoint_interval > self.max_iterations:
            msg = "checkpoint_interval cannot be greater than max_iterations"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_nested_configs(self) -> "WorkflowConfig":
        """Ensures all nested configurations are strictly validated."""
        if self.otf is not None:
            OTFConfig.model_validate(self.otf)
        if self.loop_strategy is not None:
            LoopStrategyConfig.model_validate(self.loop_strategy)
        if self.distillation is not None:
            DistillationConfig.model_validate(self.distillation)
        if self.cutout is not None:
            CutoutConfig.model_validate(self.cutout)
        return self
