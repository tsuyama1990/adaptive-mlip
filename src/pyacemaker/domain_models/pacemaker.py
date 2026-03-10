from pydantic import BaseModel, ConfigDict, Field


class PacemakerDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str


class PacemakerEmbeddingSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ndensity: int
    npot: str
    fs_parameters: list[float]
    maxwell: bool


class PacemakerBondSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    N: int = Field(alias="N")
    max_deg: int
    r0: float
    rad_base: str
    rad_parameters: list[float]


class PacemakerPotentialSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delta_spline_bins: int
    elements: list[str]
    embeddings: dict[str, PacemakerEmbeddingSchema]
    bonds: PacemakerBondSchema


class PacemakerLossSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kappa: float
    L1_coeffs: float
    L2_coeffs: float


class PacemakerFitSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loss: PacemakerLossSchema
    optimizer: str
    maxiter: int
    repulsion_sigma: float


class PacemakerBackendSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator: str
    batch_size: int
    display_step: int


class PacemakerBasePotentialSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    parameters: dict[str, dict[str, float]]


class PacemakerInputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cutoff: float
    seed: int
    data: PacemakerDataSchema
    potential: PacemakerPotentialSchema
    fit: PacemakerFitSchema
    backend: PacemakerBackendSchema
    base_potential: PacemakerBasePotentialSchema | None = None
