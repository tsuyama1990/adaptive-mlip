from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator

from pyacemaker.domain_models.defaults import FILENAME_POTENTIAL


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potential_type: str = Field(..., description="Type of potential to train")
    cutoff_radius: PositiveFloat = Field(..., description="Potential cutoff radius in Angstrom")
    max_basis_size: int = Field(..., gt=0, description="Maximum basis set size")

    # Additional Parameters for Scalability & Reproducibility
    seed: int = Field(42, description="Random seed for reproducibility")
    max_iterations: int = Field(1000, description="Maximum training iterations", gt=0)
    batch_size: int = Field(10, description="Training batch size", gt=0)
    elements: list[str] | None = Field(
        None, description="List of chemical elements in the dataset (optional optimization)"
    )

    # Mocking & Output (Audit Requirement)
    output_filename: str = Field(
        FILENAME_POTENTIAL, description="Filename for the trained potential"
    )

    @field_validator("output_filename")
    @classmethod
    def validate_filename_safe(cls, v: str) -> str:
        """Ensures filename does not contain path separators."""
        if "/" in v or "\\" in v:
            msg = "Filename cannot contain path separators"
            raise ValueError(msg)
        return v

    # Spec Section 3.3
    delta_learning: bool = Field(False, description="Use LJ baseline for delta learning")
    active_set_optimization: bool = Field(
        False, description="Use MaxVol selection for active set"
    )
    active_set_size: int | None = Field(
        None, description="Target number of structures for active set", gt=0
    )
