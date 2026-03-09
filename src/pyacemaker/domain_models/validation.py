from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from pyacemaker.domain_models.defaults import (
    DEFAULT_VALIDATION_ELASTIC_STEPS,
    DEFAULT_VALIDATION_ELASTIC_STRAIN,
    DEFAULT_VALIDATION_PHONON_DISPLACEMENT,
    DEFAULT_VALIDATION_PHONON_IMAGINARY_TOL,
    DEFAULT_VALIDATION_PHONON_SUPERCELL,
)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class FileFormatValidator:
    """Centralized validator for data file formats."""

    ALLOWED_FORMATS: ClassVar[set[str]] = {".pckl", ".xyz", ".extxyz", ".gzip"}

    @classmethod
    def validate_training_data_format(cls, data_path: Path) -> None:
        if not data_path.exists():
            msg = f"Training data not found: {data_path}"
            raise FileNotFoundError(msg)

        if data_path.suffix not in cls.ALLOWED_FORMATS:
            msg = (
                f"Invalid training data format: {data_path.suffix}. Allowed: {cls.ALLOWED_FORMATS}"
            )
            raise ValueError(msg)

        if data_path.stat().st_size == 0:
            msg = f"Training data file is empty: {data_path}"
            raise ValueError(msg)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phonon_stable: bool = Field(..., description="Whether the potential is dynamically stable")
    elastic_stable: bool = Field(..., description="Whether the potential is mechanically stable")
    c_ij: dict[str, float] = Field(..., description="Calculated elastic constants (GPa)")
    bulk_modulus: float = Field(..., description="Calculated bulk modulus (GPa)")
    plots: dict[str, str] = Field(
        default_factory=dict, description="Base64 encoded plots (keys: phonon, elastic)"
    )
    report_path: str = Field(..., description="Path to the HTML validation report")
