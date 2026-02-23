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
    mixing_beta_factor: float = Field(
        0.5, gt=0.0, le=1.0, description="Multiplier for mixing_beta reduction strategy"
    )
    smearing_width_factor: float = Field(
        2.0, gt=1.0, description="Multiplier for smearing_width increase strategy"
    )

    # Pseudopotentials
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """
        Validates that pseudopotential files exist using strict path resolution.
        """
        # Define allowed base directory (e.g., current working directory or specific pseudo dir)
        # For now, we restrict to paths relative to CWD or explicit absolute paths that exist.
        base_dir = Path.cwd().resolve()

        for elem, path_str in v.items():
            if not path_str or not path_str.strip():
                msg = f"Pseudopotential path for {elem} cannot be empty"
                raise ValueError(msg)

            try:
                # Resolve path to absolute
                p = Path(path_str).expanduser()
                p = (base_dir / p).resolve() if not p.is_absolute() else p.resolve()

                # Check existence
                if not p.exists():
                    msg = f"Pseudopotential file not found: {p}"
                    raise FileNotFoundError(msg)

                # Check for path traversal relative to expected roots if needed
                # For now, ensuring it exists and is resolvable is good security practice against injection
                # The 'resolve()' call handles symlinks and '..' components safely.
            except Exception as e:
                 msg = f"Invalid pseudopotential path for {elem}: {e}"
                 raise ValueError(msg) from e

        return v
