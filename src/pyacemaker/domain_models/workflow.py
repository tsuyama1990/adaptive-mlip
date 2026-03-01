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


class ActiveLearningThresholds(BaseModel):
    """Manages the two-tier evaluation thresholds."""
    model_config = ConfigDict(extra="forbid")

    threshold_call_dft: float = Field(..., gt=0, description="Gamma threshold to trigger a halt and check DFT.")
    threshold_add_train: float = Field(..., gt=0, description="Gamma threshold to trigger adding a structure to the training set.")
    smooth_steps: PositiveInt = Field(default=3, description="Number of consecutive steps exceeding threshold to filter thermal noise.")


class CutoutConfig(BaseModel):
    """Configuration for Phase 3: Intelligent Cutout."""
    model_config = ConfigDict(extra="forbid")

    core_radius: float = Field(..., gt=0, description="Radius of the core region (Angstrom).")
    buffer_radius: float = Field(..., gt=0, description="Thickness of the buffer region (Angstrom).")
    enable_pre_relaxation: bool = Field(default=True, description="Use MACE to pre-relax buffer atoms.")
    enable_passivation: bool = Field(default=True, description="Auto-passivate broken bonds.")


class LoopStrategyConfig(BaseModel):
    """Configuration for Active Learning Loop Strategy."""
    model_config = ConfigDict(extra="forbid")

    use_tiered_oracle: bool = Field(default=True, description="Enable MACE and DFT two-tier oracle.")
    incremental_update: bool = Field(default=True, description="Use Delta Learning for incremental updates.")
    replay_buffer_size: PositiveInt = Field(default=1000, description="Size of the Replay Buffer for incremental learning.")
    baseline_potential_type: str = Field(default="lj", description="Type of the baseline potential (e.g., lj).")


class OTFConfig(BaseModel):
    """Configuration for On-The-Fly (OTF) Active Learning loop."""
    model_config = ConfigDict(extra="forbid")

    uncertainty_threshold: float = Field(
        default=DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
        gt=0,
        description="Gamma threshold to trigger halt and retraining."
    )
    local_n_candidates: PositiveInt = Field(
        default=DEFAULT_OTF_LOCAL_N_CANDIDATES,
        description="Number of local candidates to generate around halt structure."
    )
    local_n_select: PositiveInt = Field(
        default=DEFAULT_OTF_LOCAL_N_SELECT,
        description="Number of candidates to select via active set optimization."
    )
    max_retries: PositiveInt = Field(
        default=DEFAULT_OTF_MAX_RETRIES,
        description="Maximum number of retraining attempts per iteration."
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

    otf: OTFConfig = Field(
        default_factory=OTFConfig,
        description="Configuration for OTF loop."
    )

    thresholds: ActiveLearningThresholds | None = Field(None, description="Active learning thresholds for two-tier evaluation.")
    cutout: CutoutConfig | None = Field(None, description="Configuration for cluster cutout.")
    strategy: LoopStrategyConfig = Field(default_factory=LoopStrategyConfig, description="Strategy for the learning loop.")
