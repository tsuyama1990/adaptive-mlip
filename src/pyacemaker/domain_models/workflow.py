from pydantic import BaseModel, ConfigDict, Field, PositiveInt


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
        default="state.json", description="Path to the state checkpoint file"
    )

    # New fields to avoid magic numbers
    batch_size: PositiveInt = Field(
        default=5, description="Number of structures to process in a batch"
    )
    n_candidates: PositiveInt = Field(
        default=10, description="Number of candidate structures to generate per iteration"
    )
    checkpoint_interval: PositiveInt = Field(default=1, description="Save state every N iterations")
