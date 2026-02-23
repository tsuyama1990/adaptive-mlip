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
        Validates that pseudopotential files exist and are safe.
        Enforces that relative paths are within the current working directory.
        """
        cwd = Path.cwd().resolve()
        for elem, path_str in v.items():
            if not path_str or not path_str.strip():
                msg = f"Pseudopotential path for {elem} cannot be empty"
                raise ValueError(msg)

            p = Path(path_str)

            try:
                if p.is_absolute():
                    if not p.exists():
                        msg = f"Pseudopotential file not found: {p}"
                        raise FileNotFoundError(msg)
                else:
                    # Resolve relative path
                    # Note: resolve(strict=False) resolves symlinks and '..' components.
                    # We check if the result is still inside CWD.
                    resolved_path = p.resolve()

                    if not resolved_path.is_relative_to(cwd):
                        msg = f"Path traversal detected: {path_str} resolves to {resolved_path}, which is outside {cwd}"
                        raise ValueError(msg)  # noqa: TRY301

                    if not resolved_path.exists():
                         msg = f"Pseudopotential file not found: {resolved_path}"
                         raise FileNotFoundError(msg)

            except (ValueError, OSError) as e:
                # Catch ValueError from is_relative_to (if not related) or OSError
                # Re-raise explicit ValueError for Pydantic
                msg = f"Invalid pseudopotential path {path_str}: {e}"
                raise ValueError(msg) from e

        return v
