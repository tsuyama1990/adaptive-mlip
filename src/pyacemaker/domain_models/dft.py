import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator

from pyacemaker.domain_models.defaults import (
    DEFAULT_DFT_DIAGONALIZATION,
    DEFAULT_DFT_MIXING_BETA,
    DEFAULT_DFT_MIXING_BETA_FACTOR,
    DEFAULT_DFT_SMEARING_TYPE,
    DEFAULT_DFT_SMEARING_WIDTH,
    DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
)


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal["qe", "vasp"] = Field(..., description="DFT code to use")
    functional: Literal["PBE", "LDA", "B3LYP", "optB88-vdW", "SCAN"] = Field(..., description="Exchange-correlation functional")
    kpoints_density: PositiveFloat = Field(..., description="K-points density in 1/Angstrom")
    encut: PositiveFloat = Field(..., description="Energy cutoff in eV")

    # Periodic Embedding
    embedding_buffer: float | None = Field(
        None, gt=0.0, description="Vacuum buffer for periodic embedding (Angstrom)"
    )

    # Self-healing and convergence parameters
    mixing_beta: float = Field(
        DEFAULT_DFT_MIXING_BETA, gt=0.0, le=1.0, description="Initial mixing parameter for SCF"
    )
    smearing_type: str = Field(
        DEFAULT_DFT_SMEARING_TYPE, description="Type of smearing (e.g., 'mv', 'gaussian')"
    )
    smearing_width: PositiveFloat = Field(
        DEFAULT_DFT_SMEARING_WIDTH, description="Width of smearing in eV"
    )
    diagonalization: str = Field(
        DEFAULT_DFT_DIAGONALIZATION, description="Diagonalization algorithm"
    )

    # Strategy Multipliers
    # Note: mixing_beta_factor is used to REDUCE mixing_beta (new_beta = beta * factor)
    #       smearing_width_factor is used to INCREASE smearing_width (new_width = width * factor)
    mixing_beta_factor: float = Field(
        DEFAULT_DFT_MIXING_BETA_FACTOR,
        gt=0.0,
        le=1.0,
        description="Multiplier for mixing_beta reduction strategy",
    )
    smearing_width_factor: float = Field(
        DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
        gt=1.0,
        description="Multiplier for smearing_width increase strategy",
    )

    # Pseudopotentials
    pseudopotentials: dict[str, str] = Field(
        ..., min_length=1, description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """
        Validates that pseudopotential values are strict, sanitized filenames
        without any directory traversal or path separators.
        """
        from pathlib import Path

        MAX_FILENAME_LENGTH = 255
        # Restrict to alphanumeric, underscore, minus, plus, and dots to prevent injection attacks.
        # We explicitly require that between any dots there is an allowed char, preventing '..'.
        SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\+]+(\.[a-zA-Z0-9_\-\+]+)*$")

        for elem, filename in v.items():
            if not filename or not filename.strip():
                msg = f"Pseudopotential filename for {elem} cannot be empty"
                raise ValueError(msg)

            if len(filename) > MAX_FILENAME_LENGTH:
                msg = f"Pseudopotential filename for {elem} exceeds maximum length of {MAX_FILENAME_LENGTH}"
                raise ValueError(msg)

            base_name = Path(filename).name
            if base_name != filename:
                msg = f"Pseudopotential filename for {elem} must not contain path separators: {filename}"
                raise ValueError(msg)

            if not SAFE_FILENAME_PATTERN.match(base_name):
                msg = f"Pseudopotential filename for {elem} contains invalid characters or consecutive dots: {filename}"
                raise ValueError(msg)

        return v
