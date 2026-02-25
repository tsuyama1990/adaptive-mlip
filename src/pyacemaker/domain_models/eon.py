import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.domain_models.constants import DEFAULT_EON_EXECUTABLE, DEFAULT_EON_SEED
from pyacemaker.utils.path import validate_path_safe


class EONConfig(BaseModel):
    """Configuration for EON (Adaptive Kinetic Monte Carlo)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to enable EON")
    eon_executable: str = Field(
        default_factory=lambda: os.getenv("EON_EXECUTABLE", DEFAULT_EON_EXECUTABLE),
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
        default_factory=lambda: int(os.getenv("EON_SEED", str(DEFAULT_EON_SEED))),
        description="Random seed for EON",
    )
    otf_threshold: float = Field(
        0.05, ge=0.0, description="On-The-Fly extrapolation grade threshold"
    )

    # EON specific job settings
    job_type: str = Field("akmc", description="EON job type (e.g., 'akmc', 'basin_hopping')")
    saddle_search_method: str = Field("min_mode", description="Saddle point search method")

    @field_validator("potential_path")
    @classmethod
    def validate_potential_path(cls, v: Path) -> Path:
        """Validates the potential path using secure path validation."""
        try:
            # First, standard path safety check
            validate_path_safe(v)

            # Second, existence check (EON requires it to run)
            if not v.exists():
                msg = f"Potential file does not exist: {v}"
                raise ValueError(msg)

        except ValueError as e:
            # Re-raise as ValueError for Pydantic to catch
            raise ValueError(str(e)) from e

        return v
