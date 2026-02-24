from enum import StrEnum

from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExplorationPolicy(StrEnum):
    COLD_START = "cold_start"
    RANDOM_RATTLE = "random_rattle"
    STRAIN = "strain"
    DEFECTS = "defects"


class StrainMode(StrEnum):
    VOLUME = "volume"
    SHEAR = "shear"
    MIXED = "mixed"


class StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(..., min_length=1, description="List of elements in the system")
    supercell_size: list[int] = Field(
        ..., min_length=3, max_length=3, description="Supercell size [nx, ny, nz]"
    )

    # Exploration Policy Configuration
    policy_name: ExplorationPolicy = Field(
        default=ExplorationPolicy.COLD_START, description="Exploration policy to use"
    )
    rattle_stdev: float = Field(
        default=0.1, ge=0.0, description="Standard deviation for random rattle (Angstrom)"
    )
    strain_mode: StrainMode = Field(
        default=StrainMode.VOLUME, description="Mode for strain application"
    )
    strain_magnitude: float = Field(
        default=0.05, ge=0.0, description="Magnitude of strain to apply (e.g., 0.05 for 5%)"
    )
    vacancy_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Rate of vacancies to introduce"
    )
    num_structures: int = Field(default=1, ge=1, description="Number of structures to generate")

    # Local Active Learning Settings
    local_extraction_radius: float = Field(
        default=6.0,
        gt=0.0,
        description="Radius for extracting local clusters around high uncertainty atoms (Angstrom)",
    )
    local_buffer_radius: float = Field(
        default=4.0,
        ge=0.0,
        description="Buffer radius added to extraction for force masking (Angstrom)",
    )

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "Elements list cannot be empty"
            raise ValueError(msg)

        # Check for duplicates
        if len(v) != len(set(v)):
            msg = "Elements list cannot contain duplicates"
            raise ValueError(msg)

        valid_symbols = set(chemical_symbols)
        for el in v:
            if el not in valid_symbols:
                msg = f"Invalid chemical symbol: {el}"
                raise ValueError(msg)
        return v

    @field_validator("supercell_size")
    @classmethod
    def validate_supercell(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            msg = "Supercell dimensions must be positive integers"
            raise ValueError(msg)
        return v
