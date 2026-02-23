from pathlib import Path

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

    code: str = Field(..., description="DFT code to use")
    functional: str = Field(..., description="Exchange-correlation functional")
    kpoints_density: PositiveFloat = Field(..., description="K-points density in 1/Angstrom")
    encut: PositiveFloat = Field(..., description="Energy cutoff in eV")

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
    diagonalization: str = Field(DEFAULT_DFT_DIAGONALIZATION, description="Diagonalization algorithm")

    # Strategy Multipliers
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
                # It also resolves symlinks.
                resolved_path = p.resolve(strict=True)

                # Check path traversal
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
