from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EONConfig(BaseModel):
    """Configuration for EON (Adaptive Kinetic Monte Carlo)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to enable EON")
    eon_executable: str = Field("eonclient", description="Path to EON executable")
    potential_path: Path = Field(..., description="Path to the potential file")
    temperature: float = Field(300.0, ge=0.0, description="Temperature in Kelvin")
    akmc_steps: int = Field(100, ge=1, description="Number of aKMC steps to run")
    supercell: list[int] = Field(
        default_factory=lambda: [1, 1, 1],
        min_length=3,
        max_length=3,
        description="Supercell dimensions",
    )
    # Additional EON parameters could go here
    mpi_command: str | None = Field(None, description="MPI command prefix (e.g., 'mpirun -np 4')")
