from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EONConfig(BaseModel):
    """
    Configuration for EON (Adaptive Kinetic Monte Carlo) simulations.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Enable EON/kMC calculation")
    temperature: float = Field(..., gt=0, description="Temperature for kMC simulation (K)")

    # EON execution parameters
    search_method: str = Field("akmc", description="Search method (e.g., akmc, min_mode)")
    job_type: str = Field("akmc", description="Job type for config.ini")
    potential_type: str = Field("ext", description="Potential type (e.g., ext, lammps)")
    random_seed: int = Field(12345, description="Random seed for EON")

    # Optional override for potential path, otherwise uses the trained potential
    potential_path: Path | None = Field(None, description="Path to potential file (optional override)")

    # Parallelization
    num_replicas: int = Field(1, ge=1, description="Number of parallel replicas")

    # KMC Parameters
    confidence: float = Field(0.99, gt=0, lt=1, description="Confidence level for transition search")
