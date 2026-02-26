from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class AtomStructure(BaseModel):
    """
    Wrapper around ASE Atoms object with additional metadata for the pipeline.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    atoms: Atoms = Field(..., description="ASE Atoms object")
    energy: float | None = Field(None, description="Potential Energy (eV)")
    forces: list[list[float]] | None = Field(None, description="Forces (eV/Angstrom)")
    uncertainty: float | None = Field(None, description="Uncertainty metric")
    provenance: str = Field(..., description="Source of the structure (e.g. DIRECT_SAMPLING)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
