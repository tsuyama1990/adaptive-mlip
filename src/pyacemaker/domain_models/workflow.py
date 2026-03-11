from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from pyacemaker.domain_models.defaults import (
    DEFAULT_ACTIVE_LEARNING_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DATA_DIR,
    DEFAULT_DISTILLATION_SAMPLING_STRUCTURES,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OTF_LOCAL_N_CANDIDATES,
    DEFAULT_OTF_LOCAL_N_SELECT,
    DEFAULT_OTF_MAX_RETRIES,
    DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
    DEFAULT_POTENTIALS_DIR,
    DEFAULT_STATE_FILE,
)


class OTFConfig(BaseModel):
    """Configuration for On-The-Fly (OTF) Active Learning loop."""

    model_config = ConfigDict(extra="forbid")

    uncertainty_threshold: float = Field(
        default=DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
        gt=0,
        description="Gamma threshold to trigger halt and retraining.",
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


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="Threshold where MACE is confident")
    sampling_structures_per_system: int = DEFAULT_DISTILLATION_SAMPLING_STRUCTURES


class ActiveLearningThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold_call_dft: float = Field(0.05, description="Criterion to halt MD and call DFT")
    threshold_add_train: float = Field(
        0.02, description="Criterion to select atoms to add to training set"
    )
    smooth_steps: int = Field(
        3, description="Consecutive steps required to exceed threshold to exclude thermal noise"
    )


class CutoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core_radius: float = Field(4.0, description="Radius for Force Weight 1.0")
    buffer_radius: float = Field(3.0, description="Thickness of additional relaxation buffer layer")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"
    pre_relax_fmax: float = Field(0.05, description="Force tolerance for pre-relaxation")
    pre_relax_steps: int = Field(50, description="Maximum steps for pre-relaxation")

    @model_validator(mode="after")
    def validate_radii(self) -> "CutoutConfig":
        if self.core_radius <= 0:
            msg = "core_radius must be positive"
            raise ValueError(msg)
        if self.buffer_radius < 0:
            msg = "buffer_radius must be non-negative"
            raise ValueError(msg)
        return self


class LoopStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(
        500, description="Number of past data points to retain to prevent catastrophic forgetting"
    )
    baseline_potential_type: str = Field("LJ", description="Baseline physical potential (e.g., LJ)")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: PositiveInt = Field(..., description="Maximum number of active learning cycles")
    convergence_energy: float = Field(
        default=0.001, gt=0, description="Energy convergence criteria in eV/atom"
    )
    convergence_force: float = Field(
        default=0.01, gt=0, description="Force convergence criteria in eV/Angstrom"
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

    distillation: DistillationConfig = Field(
        default_factory=DistillationConfig, description="Configuration for Zero-Shot Distillation."
    )

    cutout: CutoutConfig = Field(
        default_factory=CutoutConfig, description="Configuration for intelligent cluster cutout."
    )

    loop_strategy: LoopStrategyConfig = Field(
        default_factory=LoopStrategyConfig,
        description="Configuration for next generation learning strategy.",
    )
