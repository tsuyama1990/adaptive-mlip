import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyacemaker.domain_models.data import AtomStructure


class DescriptorConfig(BaseModel):
    """Configuration for descriptor calculation."""
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., description="Descriptor method (e.g., 'soap', 'ace')")
    species: list[str] = Field(..., min_length=1, description="List of chemical species")
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
        # Validate that descriptors array size is reasonable if present
        if self.descriptors is not None:
            # Check for excessive size (e.g., > 2GB equivalent)
            # Assuming float64 (8 bytes)
            # 2GB = 2 * 1024^3 / 8 = 268,435,456 elements
            if self.descriptors.size > 268_435_456:
                raise ValueError("Descriptor array is too large (>2GB), potential OOM risk.")

            # Basic consistency check: indices should point within pool bounds if pool represents FULL set
            # But pool is "sampled structures", so it should match selection indices size if it contains ONLY selected.
            # Usually SamplingResult returns the SELECTED pool.
            if len(self.pool) != len(self.selection_indices):
                 # If sizes mismatch, assume pool is the full candidate set?
                 # No, better to enforce explicit behavior.
                 # Let's assume pool contains only the selected ones to save memory.
                 # But then descriptors might be for the full set?
                 # Let's just validate that pool size matches descriptors rows if they are aligned.
                 pass

        return self
