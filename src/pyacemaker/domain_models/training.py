from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potential_type: str = Field(..., description="Type of potential to train")
    cutoff_radius: PositiveFloat = Field(..., description="Potential cutoff radius in Angstrom")
    max_basis_size: int = Field(..., gt=0, description="Maximum basis set size")

    # Mocking & Output (Audit Requirement)
    output_filename: str = Field(
        "potential.yace", description="Filename for the trained potential"
    )

    # Spec Section 3.3
    delta_learning: bool = Field(False, description="Use LJ baseline for delta learning")
    active_set_optimization: bool = Field(
        False, description="Use MaxVol selection for active set"
    )
