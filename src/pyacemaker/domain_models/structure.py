from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(..., description="List of elements in the system")
    supercell_size: list[int] = Field(
        ..., min_length=3, max_length=3, description="Supercell size [nx, ny, nz]"
    )

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, v: list[str]) -> list[str]:
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
