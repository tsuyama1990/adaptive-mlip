from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EONConfig(BaseModel):
    """Configuration for EON (Adaptive Kinetic Monte Carlo)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether to enable EON")
    eon_executable: str = Field("eonclient", description="Path to EON executable")
    potential_path: Path = Field(..., description="Path to the potential file")
    temperature: float = Field(300.0, ge=0.0, description="Temperature in Kelvin")
    akmc_steps: int = Field(100, ge=1, description="Number of aKMC steps to run")
    random_seed: int = Field(12345, description="Random seed for EON")
    otf_threshold: float = Field(5.0, gt=0.0, description="Gamma threshold for OTF halt")
    supercell: list[int] = Field(
        default_factory=lambda: [1, 1, 1],
        min_length=3,
        max_length=3,
        description="Supercell dimensions",
    )
    mpi_command: str | None = Field(None, description="MPI command prefix (e.g., 'mpirun -np 4')")

    @field_validator("potential_path")
    @classmethod
    def validate_potential_path(cls, v: Path) -> Path:
        # Only validate existence if we are actually running (enabled logic checked elsewhere,
        # but pure schema validation runs on load).
        # We should check if it exists if it's provided.
        # But during dry-run, file might not exist yet?
        # Specification says "validate that potential_path exists or is accessible".
        # We'll rely on resolve checking existence for strictness, but allows non-strict for future artifacts?
        # Let's enforce existence unless it's a dry run context (which Pydantic doesn't know).
        # However, potential usually exists BEFORE EON runs.
        # We will assume it must exist at config load time for safety.
        if not v.exists():
             msg = f"Potential file does not exist: {v}"
             raise ValueError(msg)
        return v
