
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator, model_validator

from pyacemaker.domain_models.defaults import (
    DEFAULT_DELTA_SPLINE_BINS,
    DEFAULT_DISPLAY_STEP,
    DEFAULT_EVALUATOR,
    DEFAULT_PACEMAKER_EMBEDDING_TYPE,
    DEFAULT_PACEMAKER_LOSS_KAPPA,
    DEFAULT_PACEMAKER_LOSS_L1,
    DEFAULT_PACEMAKER_LOSS_L2,
    DEFAULT_PACEMAKER_MAX_DEG,
    DEFAULT_PACEMAKER_NDENSITY,
    DEFAULT_PACEMAKER_OPTIMIZER,
    DEFAULT_PACEMAKER_R0,
    DEFAULT_PACEMAKER_RAD_BASE,
    DEFAULT_PACEMAKER_REPULSION_SIGMA,
    DEFAULT_TRAINING_BATCH_SIZE,
    DEFAULT_TRAINING_MAX_ITERATIONS,
    FILENAME_POTENTIAL,
)


class PacemakerConfig(BaseModel):
    """Specific configuration for Pacemaker training."""

    model_config = ConfigDict(extra="allow")

    # Embedding settings
    embedding_type: str = Field(
        DEFAULT_PACEMAKER_EMBEDDING_TYPE, description="Type of embedding function"
    )
    fs_parameters: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.5],
        description="Parameters for FinnisSinclair embedding",
    )
    ndensity: int = Field(
        DEFAULT_PACEMAKER_NDENSITY, description="Density expansion order", gt=0
    )

    # Bond settings
    rad_base: str = Field(DEFAULT_PACEMAKER_RAD_BASE, description="Radial basis function type")
    rad_parameters: list[float] = Field(
        default_factory=lambda: [1.0], description="Radial basis parameters"
    )
    max_deg: int = Field(
        DEFAULT_PACEMAKER_MAX_DEG, description="Maximum degree of expansion", gt=0
    )
    r0: float = Field(DEFAULT_PACEMAKER_R0, description="Radial cutoff shift", gt=0)

    # Loss settings
    loss_kappa: float = Field(
        DEFAULT_PACEMAKER_LOSS_KAPPA, description="Kappa parameter for loss function", ge=0
    )
    loss_l1_coeffs: float = Field(
        DEFAULT_PACEMAKER_LOSS_L1, description="L1 regularization coefficient", ge=0
    )
    loss_l2_coeffs: float = Field(
        DEFAULT_PACEMAKER_LOSS_L2, description="L2 regularization coefficient", ge=0
    )
    repulsion_sigma: float = Field(
        DEFAULT_PACEMAKER_REPULSION_SIGMA, description="Repulsion sigma", gt=0
    )

    # Optimizer settings
    optimizer: str = Field(DEFAULT_PACEMAKER_OPTIMIZER, description="Optimization algorithm")

    # Advanced Settings (Moved from hardcoded values)
    delta_spline_bins: int = Field(
        DEFAULT_DELTA_SPLINE_BINS, description="Number of bins for delta spline", gt=0
    )
    evaluator: str = Field(DEFAULT_EVALUATOR, description="Backend evaluator for potential")
    display_step: int = Field(
        DEFAULT_DISPLAY_STEP, description="Frequency of logging during training", gt=0
    )


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    potential_type: str = Field(..., description="Type of potential to train")
    cutoff_radius: PositiveFloat = Field(..., description="Potential cutoff radius in Angstrom")
    max_basis_size: int = Field(..., gt=0, description="Maximum basis set size")

    # Additional Parameters for Scalability & Reproducibility
    seed: int = Field(42, description="Random seed for reproducibility")
    max_iterations: int = Field(
        DEFAULT_TRAINING_MAX_ITERATIONS, description="Maximum training iterations", gt=0
    )
    batch_size: int = Field(
        DEFAULT_TRAINING_BATCH_SIZE, description="Training batch size", gt=0
    )
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
