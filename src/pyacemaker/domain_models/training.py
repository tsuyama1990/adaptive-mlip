from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from pyacemaker.domain_models.defaults import FILENAME_POTENTIAL


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potential_type: str = Field(..., description="Type of potential to train")
    cutoff_radius: PositiveFloat = Field(..., description="Potential cutoff radius in Angstrom")
    max_basis_size: int = Field(..., gt=0, description="Maximum basis set size")

    # Mocking & Output (Audit Requirement)
    output_filename: str = Field(
        FILENAME_POTENTIAL, description="Filename for the trained potential"
    )

    # Spec Section 3.3
    delta_learning: bool = Field(False, description="Use LJ baseline for delta learning")
    active_set_optimization: bool = Field(
        False, description="Use MaxVol selection for active set"
    )
    active_set_size: int | None = Field(
        None, description="Target number of structures for active set", gt=0
    )
