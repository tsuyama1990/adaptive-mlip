from enum import StrEnum
from typing import Literal

from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExplorationPolicy(StrEnum):
    COLD_START = "cold_start"
    RANDOM_RATTLE = "random_rattle"
    STRAIN = "strain"
    DEFECTS = "defects"


class StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(..., min_length=1, description="List of elements in the system")
    supercell_size: list[int] = Field(
        ..., min_length=3, max_length=3, description="Supercell size [nx, ny, nz]"
    )

    # Exploration Policy Configuration
    policy_name: ExplorationPolicy = Field(
        default=ExplorationPolicy.COLD_START, description="Active exploration policy"
    )

    # Policy Parameters
    rattle_stdev: float = Field(
        0.1, ge=0.0, description="Standard deviation for random rattle (Angstrom)"
    )
    strain_mode: Literal["volume", "shear", "full"] = Field(
        "full", description="Mode of strain application"
    )
    vacancy_rate: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Fraction of atoms to remove as vacancies (0.0 - 1.0)",
    )

    # Adaptive Exploration Policy Parameters (Spec Section 3.1)
    adaptive_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="MD/MC Ratio (0.0 = Pure MD, 1.0 = Pure MC)"
    )
    defect_density: float = Field(
        0.0, ge=0.0, description="Concentration of defects to introduce (Legacy)"
    )
    strain_range: float = Field(
        0.05, ge=0.0, description="Range of strain for elastic sampling (e.g. 0.05 = +/- 5%)"
    )

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "Elements list cannot be empty"
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
