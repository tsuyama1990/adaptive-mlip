from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from pyacemaker.constants import (
    DEFAULT_ACTIVE_LEARNING_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DATA_DIR,
    DEFAULT_N_CANDIDATES,
    DEFAULT_POTENTIALS_DIR,
    DEFAULT_STATE_FILE,
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
