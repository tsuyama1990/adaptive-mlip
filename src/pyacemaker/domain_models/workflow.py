from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from pyacemaker.domain_models.defaults import (
    DEFAULT_ACTIVE_LEARNING_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DATA_DIR,
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
    enable: bool = Field(False, description="Enable Phase 1 Zero-Shot Distillation")
    mace_model_path: str = Field(
        default="MACE-MP-0", description="Path or name of MACE foundation model"
    )
    uncertainty_threshold: float = Field(
        default=0.1,
        description="Uncertainty threshold for retaining structures",
        json_schema_extra={"env": "DISTILLATION_THRESHOLD"},
    )
    sampling_counts: PositiveInt = Field(
        default=1000, description="Number of structures to sample via DIRECT"
    )


class ActiveLearningThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold_call_dft: float = Field(
        default=2.0, description="Threshold to trigger halt and call DFT"
    )
    threshold_add_train: float = Field(
        default=5.0, description="Threshold to add data to training set (epicentre)"
    )
    smooth_steps: PositiveInt = Field(
        default=3, description="Consecutive steps required to filter thermal noise"
    )


class CutoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    core_radius: float = Field(default=4.0, description="Radius for core atoms (force_weight=1.0)")
    buffer_radius: float = Field(
        default=8.0, description="Radius for buffer atoms (force_weight=0.0)"
    )
    enable_pre_relaxation: bool = Field(
        default=True, description="Enable pre-relaxation of buffer using MACE"
    )
    enable_passivation: bool = Field(
        default=True, description="Enable auto-passivation of broken bonds"
    )


class LoopStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_tiered_oracle: bool = Field(default=True, description="Route queries through TieredOracle")
    incremental_update: bool = Field(
        default=True, description="Use Delta Learning for incremental updates"
    )
    replay_buffer_size: PositiveInt = Field(
        default=1000, description="Max size of replay buffer for historical data"
    )
    baseline_potential_type: str = Field(
        default="LJ", description="Type of baseline potential (e.g., LJ)"
    )


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

    distillation: DistillationConfig = Field(
        default_factory=DistillationConfig, description="Configuration for Distillation Phase."
    )
    thresholds: ActiveLearningThresholds = Field(
        default_factory=ActiveLearningThresholds,
        description="Configuration for Active Learning Thresholds.",
    )
    cutout: CutoutConfig = Field(
        default_factory=CutoutConfig,
        description="Configuration for Cluster Cutout and Passivation.",
    )
    strategy: LoopStrategyConfig = Field(
        default_factory=LoopStrategyConfig, description="Configuration for Loop Strategy."
    )
    otf: OTFConfig = Field(default_factory=OTFConfig, description="Configuration for OTF loop.")
