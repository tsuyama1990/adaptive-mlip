from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from pyacemaker.domain_models.defaults import (
    DEFAULT_VALIDATION_ELASTIC_STEPS,
    DEFAULT_VALIDATION_ELASTIC_STRAIN,
    DEFAULT_VALIDATION_PHONON_DISPLACEMENT,
    DEFAULT_VALIDATION_PHONON_IMAGINARY_TOL,
    DEFAULT_VALIDATION_PHONON_SUPERCELL,
)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    phonon_supercell: list[int] = Field(
        default=DEFAULT_VALIDATION_PHONON_SUPERCELL,
        description="Supercell dimensions for phonon calculation (e.g., [2, 2, 2])",
    )
    phonon_displacement: PositiveFloat = Field(
        default=DEFAULT_VALIDATION_PHONON_DISPLACEMENT,
        description="Atomic displacement for phonon finite difference method (Angstrom)",
    )
    phonon_imaginary_tol: float = Field(
        default=DEFAULT_VALIDATION_PHONON_IMAGINARY_TOL,
        description="Tolerance for imaginary frequencies (e.g. -0.05 THz)",
    )
    elastic_strain: PositiveFloat = Field(
        default=DEFAULT_VALIDATION_ELASTIC_STRAIN,
        description="Maximum strain for elastic constant calculation",
    )
    elastic_steps: PositiveInt = Field(
        default=DEFAULT_VALIDATION_ELASTIC_STEPS,
        description="Number of strain steps for fitting",
    )


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    phonon_stable: bool = Field(..., description="Whether the potential is dynamically stable")
    elastic_stable: bool = Field(..., description="Whether the potential is mechanically stable")
    c_ij: dict[str, float] = Field(..., description="Calculated elastic constants (GPa)")
    bulk_modulus: float = Field(..., description="Calculated bulk modulus (GPa)")
    plots: dict[str, str] = Field(
        default_factory=dict, description="Base64 encoded plots (keys: phonon, elastic)"
    )
    report_path: str = Field(..., description="Path to the HTML validation report")
