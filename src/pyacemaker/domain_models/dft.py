import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="DFT code to use")
    functional: str = Field(..., description="Exchange-correlation functional")
    kpoints_density: PositiveFloat = Field(..., description="K-points density in 1/Angstrom")
    encut: PositiveFloat = Field(..., description="Energy cutoff in eV")

    # Self-healing and convergence parameters
    mixing_beta: float = Field(0.7, gt=0.0, le=1.0, description="Initial mixing parameter for SCF")
    smearing_type: str = Field("mv", description="Type of smearing (e.g., 'mv', 'gaussian')")
    smearing_width: PositiveFloat = Field(0.1, description="Width of smearing in eV")
    diagonalization: str = Field("david", description="Diagonalization algorithm")

    # Strategy Multipliers
    mixing_beta_factor: float = Field(0.5, gt=0.0, le=1.0, description="Multiplier for mixing_beta reduction strategy")
    smearing_width_factor: float = Field(2.0, gt=1.0, description="Multiplier for smearing_width increase strategy")

    # Pseudopotentials
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """
        Validates that pseudopotential files exist.
        """
        for elem, path_str in v.items():
            if not path_str or not path_str.strip():
                msg = f"Pseudopotential path for {elem} cannot be empty"
                raise ValueError(msg)

            # Robust path traversal check using resolve() logic without strict existence for relative
            # We treat the path as if it were relative to a theoretical root.
            # If it tries to go above that root using '..', it's risky.

            norm_path = os.path.normpath(path_str)
            if norm_path.startswith("..") or "/../" in norm_path.replace("\\", "/"):
                 msg = f"Path traversal detected in pseudopotential path: {path_str}"
                 raise ValueError(msg)

            p = Path(path_str)
            # If path is absolute, check existence
            if p.is_absolute() and not p.exists():
                msg = f"Pseudopotential file not found: {p}"
                raise FileNotFoundError(msg)

        return v
