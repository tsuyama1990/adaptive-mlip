from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potential_type: str = Field(..., description="Type of potential to train")
    cutoff_radius: PositiveFloat = Field(..., description="Potential cutoff radius in Angstrom")
    max_basis_size: int = Field(..., gt=0, description="Maximum basis set size")
