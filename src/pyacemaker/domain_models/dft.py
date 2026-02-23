from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="DFT code to use")
    functional: str = Field(..., description="Exchange-correlation functional")
    kpoints_density: PositiveFloat = Field(..., description="K-points density in 1/Angstrom")
    encut: PositiveFloat = Field(..., description="Energy cutoff in eV")

    # Periodic Embedding
    embedding_buffer: float | None = Field(
        None, gt=0.0, description="Vacuum buffer for periodic embedding (Angstrom)"
    )

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
        Validates that pseudopotential files exist and are safe.
        Enforces that relative paths are within the current working directory.
        Uses resolve(strict=True) to ensure existence and handle symlinks safely.
        """
        cwd = Path.cwd().resolve()
        for elem, path_str in v.items():
            if not path_str or not path_str.strip():
                msg = f"Pseudopotential path for {elem} cannot be empty"
                raise ValueError(msg)

            try:
                p = Path(path_str)
                # resolve(strict=True) will raise FileNotFoundError if file doesn't exist.
                # It also resolves symlinks to their target.
                resolved_path = p.resolve(strict=True)

                # Explicitly disallow symlinks for security
                if p.is_symlink():
                     msg = f"Symlinks are not allowed for pseudopotentials: {path_str}"
                     raise ValueError(msg)  # noqa: TRY301

                # Check path traversal
                # Ensure we are comparing absolute paths
                if not resolved_path.is_absolute():
                    # Should be absolute after resolve(), but defensive check
                    resolved_path = resolved_path.absolute()

                if not resolved_path.is_relative_to(cwd):
                    msg = f"Path traversal detected: {path_str} resolves to {resolved_path}, which is outside {cwd}"
                    raise ValueError(msg)  # noqa: TRY301

                # Check if it's a file
                if not resolved_path.is_file():
                    msg = f"Pseudopotential path is not a file: {resolved_path}"
                    raise ValueError(msg)  # noqa: TRY301

            except FileNotFoundError as e:
                # Re-raise with informative message for Pydantic
                msg = f"Pseudopotential file not found: {path_str}"
                raise ValueError(msg) from e
            except (ValueError, OSError) as e:
                # Catch ValueError from is_relative_to (if not related) or OSError
                msg = f"Invalid pseudopotential path {path_str}: {e}"
                raise ValueError(msg) from e

        return v
