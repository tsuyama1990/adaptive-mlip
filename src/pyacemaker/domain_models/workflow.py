from enum import StrEnum

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


class WorkflowStep(StrEnum):
    DIRECT_SAMPLING = "direct_sampling"
    ACTIVE_LEARNING = "active_learning"
    MACE_FINETUNE = "mace_finetune"
    SURROGATE_SAMPLING = "surrogate_sampling"
    SURROGATE_LABELING = "surrogate_labeling"
    PACEMAKER_BASE = "pacemaker_base"
    DELTA_LEARNING = "delta_learning"


class OTFConfig(BaseModel):
    """Configuration for On-The-Fly (OTF) Active Learning loop."""
    model_config = ConfigDict(extra="allow")

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
    model_config = ConfigDict(extra="allow")

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
