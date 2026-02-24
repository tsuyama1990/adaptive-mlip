from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pyacemaker.domain_models.defaults import (
    DEFAULT_VALIDATION_ELASTIC_STRAIN,
    DEFAULT_VALIDATION_PHONON_DISPLACEMENT,
    DEFAULT_VALIDATION_PHONON_SUPERCELL,
    DEFAULT_VALIDATION_PHONON_SYMPREC,
)


class ValidationStatus(StrEnum):
    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class PhononConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supercell_size: tuple[int, int, int] = Field(
        default=DEFAULT_VALIDATION_PHONON_SUPERCELL,
        description="Supercell size for phonon calculation [nx, ny, nz]"
    )
    displacement: float = Field(
        default=DEFAULT_VALIDATION_PHONON_DISPLACEMENT,
        gt=0.0,
        description="Atomic displacement distance for force constants"
    )
    symprec: float = Field(
        default=DEFAULT_VALIDATION_PHONON_SYMPREC,
        gt=0.0,
        description="Symmetry tolerance"
    )


class ElasticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strain_magnitude: float = Field(
        default=DEFAULT_VALIDATION_ELASTIC_STRAIN,
        gt=0.0,
        description="Magnitude of strain for elastic constant calculation"
    )


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phonon: PhononConfig = Field(
        default_factory=PhononConfig,
        description="Configuration for Phonon validation"
    )
    elastic: ElasticConfig = Field(
        default_factory=ElasticConfig,
        description="Configuration for Elastic validation"
    )
    enabled: bool = Field(
        default=True,
        description="Whether to run validation"
    )


class PhononResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_imaginary_modes: bool
    band_structure_path: Path | None = None
    dos_path: Path | None = None
    status: ValidationStatus


class ElasticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    c_ij: dict[str, float]
    bulk_modulus: float
    is_mechanically_stable: bool
    status: ValidationStatus


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phonon: PhononResult | None = None
    elastic: ElasticResult | None = None
    overall_status: ValidationStatus
    report_path: Path | None = None
