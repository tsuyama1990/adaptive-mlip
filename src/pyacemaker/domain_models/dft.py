from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(default="quantum_espresso", description="DFT code to use")
    functional: str = Field(default="PBE", description="Exchange-correlation functional")
    kpoints_density: PositiveFloat = Field(
        default=0.04, description="K-points density in 1/Angstrom"
    )
    encut: PositiveFloat = Field(..., description="Energy cutoff in eV")
