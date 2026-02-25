import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.utils.path import validate_path_safe


class EONConfig(BaseModel):
    """Configuration for EON (Adaptive Kinetic Monte Carlo)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to enable EON")
    eon_executable: str = Field(
        default_factory=lambda: os.getenv("EON_EXECUTABLE", "eonclient"),
        description="Path to EON executable",
    )
    potential_path: Path = Field(..., description="Path to the potential file")
    temperature: float = Field(300.0, ge=0.0, description="Temperature in Kelvin")
    akmc_steps: int = Field(100, ge=1, description="Number of aKMC steps to run")
    supercell: list[int] = Field(
        default_factory=lambda: [1, 1, 1],
        min_length=3,
        max_length=3,
        description="Supercell dimensions",
    )
    mpi_command: str | None = Field(None, description="MPI command prefix (e.g., 'mpirun -np 4')")
    random_seed: int = Field(
        default_factory=lambda: int(os.getenv("EON_SEED", "12345")),
        description="Random seed for EON",
    )
    otf_threshold: float = Field(0.05, ge=0.0, description="On-The-Fly extrapolation grade threshold")

    @field_validator("potential_path")
    @classmethod
    def validate_potential_path(cls, v: Path) -> Path:
        # Validate existence using safe path validator
        # This checks for traversal and existence (if exists)
        try:
            # We validate it strictly if we can resolve it,
            # but usually it should exist before config load for strictness.
            # However, validate_path_safe raises if outside allowed root.
            # Here we just check existence primarily.
            if not v.exists():
                msg = f"Potential file does not exist: {v}"
                raise ValueError(msg)

            # Additional safety check
            validate_path_safe(v)

        except ValueError as e:
            # Re-raise as ValueError for Pydantic
            raise ValueError(str(e)) from e

        return v
