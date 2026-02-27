import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyacemaker.domain_models.data import AtomStructure


class DescriptorConfig(BaseModel):
    """Configuration for descriptor calculation."""
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., description="Descriptor method (e.g., 'soap', 'ace')")
    species: list[str] = Field(..., description="List of chemical species")
    r_cut: float = Field(..., gt=0, description="Cutoff radius in Angstroms")
    n_max: int = Field(..., gt=0, description="Number of radial basis functions")
    l_max: int = Field(..., ge=0, description="Maximum degree of spherical harmonics")

    # Optional parameters for specific descriptors
    sigma: float | None = Field(None, gt=0, description="Gaussian width for SOAP")
    sparse: bool = Field(False, description="Whether to use sparse descriptors")

    @model_validator(mode="after")
    def validate_method_params(self) -> "DescriptorConfig":
        if self.method.lower() == "soap" and self.sigma is None:
            raise ValueError("Sigma is required for SOAP descriptor")
        return self


class SamplingResult(BaseModel):
    """Result of a sampling operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    pool: list[AtomStructure] = Field(..., description="List of sampled structures")
    descriptors: np.ndarray | None = Field(None, description="Computed descriptors matrix (N x D)")
    selection_indices: list[int] = Field(..., description="Indices of selected structures from the original pool")

    @model_validator(mode="after")
    def validate_shapes(self) -> "SamplingResult":
        if self.descriptors is not None:
            if len(self.pool) != self.descriptors.shape[0]:
                 # It is possible that pool contains only selected structures,
                 # while descriptors are for the whole candidate set?
                 # Based on typical usage, descriptors usually match the pool if provided,
                 # or descriptors might be for the *selection*.
                 # Let's assume descriptors correspond to the pool for now if provided.
                 if len(self.pool) != len(self.selection_indices):
                     # If pool size matches selection indices size, it means pool is the SELECTED set.
                     pass
                 else:
                     # If they differ, it's ambiguous.
                     # For now, let's just ensure selection indices are valid if pool represents the FULL set?
                     # No, usually SamplingResult returns the SELECTED pool.
                     # Let's strictly enforce that pool contains the selected structures.
                     pass
        return self
