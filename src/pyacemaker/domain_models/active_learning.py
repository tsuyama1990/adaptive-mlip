from pydantic import BaseModel, ConfigDict, Field

class DescriptorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: str = Field(..., description="Descriptor method (e.g. soap)")
    species: list[str] = Field(..., description="Species involved")
    r_cut: float = Field(..., gt=0.0, description="Cutoff radius")
    n_max: int = Field(..., gt=0, description="Radial basis functions")
    l_max: int = Field(..., ge=0, description="Angular basis functions")
    sigma: float | None = Field(None, gt=0.0, description="Gaussian smearing")
    sparse: bool = Field(False, description="Whether to use sparse descriptor matrix")
