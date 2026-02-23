
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator, model_validator

from pyacemaker.domain_models.defaults import FILENAME_POTENTIAL


class PacemakerConfig(BaseModel):
    """Specific configuration for Pacemaker training."""

    model_config = ConfigDict(extra="forbid")

    # Embedding settings
    embedding_type: str = Field("FinnisSinclair", description="Type of embedding function")
    fs_parameters: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.5],
        description="Parameters for FinnisSinclair embedding",
    )
    ndensity: int = Field(2, description="Density expansion order", gt=0)

    # Bond settings
    rad_base: str = Field("Chebyshev", description="Radial basis function type")
    rad_parameters: list[float] = Field(
        default_factory=lambda: [1.0], description="Radial basis parameters"
    )
    max_deg: int = Field(6, description="Maximum degree of expansion", gt=0)
    r0: float = Field(1.5, description="Radial cutoff shift", gt=0)

    # Loss settings
    loss_kappa: float = Field(0.3, description="Kappa parameter for loss function", ge=0)
    loss_l1_coeffs: float = Field(1e-8, description="L1 regularization coefficient", ge=0)
    loss_l2_coeffs: float = Field(1e-8, description="L2 regularization coefficient", ge=0)
    repulsion_sigma: float = Field(0.05, description="Repulsion sigma", gt=0)

    # Optimizer settings
    optimizer: str = Field("BFGS", description="Optimization algorithm")


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

    pacemaker: PacemakerConfig = Field(
        default_factory=PacemakerConfig, description="Detailed Pacemaker configuration"
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

    @model_validator(mode="after")
    def validate_active_set_size(self) -> "TrainingConfig":
        """Ensures active_set_size is set if active_set_optimization is enabled."""
        if self.active_set_optimization and self.active_set_size is None:
            msg = "active_set_size must be set when active_set_optimization is True"
            raise ValueError(msg)
        return self
