from pydantic import BaseModel, ConfigDict, Field


class StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(..., description="List of elements in the system")
    supercell_size: list[int] = Field(
        default=[1, 1, 1], min_length=3, max_length=3, description="Supercell size [nx, ny, nz]"
    )
