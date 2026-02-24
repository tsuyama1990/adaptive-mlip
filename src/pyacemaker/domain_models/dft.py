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

    @staticmethod
    def _validate_single_pp(elem: str, path_str: str) -> None:
        """Helper to validate a single pseudopotential file."""
        import os

        if not path_str or not path_str.strip():
            msg = f"Pseudopotential path for {elem} cannot be empty"
            raise ValueError(msg)

        try:
            p = Path(path_str)
            resolved_path = p.resolve(strict=True)

            if p.is_symlink():
                msg = f"Symlinks are not allowed for pseudopotentials: {path_str}"
                raise ValueError(msg)

            if not resolved_path.is_file():
                msg = f"Pseudopotential path is not a file: {resolved_path}"
                raise ValueError(msg)

            if not os.access(resolved_path, os.R_OK):
                msg = f"Pseudopotential file is not readable: {resolved_path}"
                raise PermissionError(msg)

        except FileNotFoundError as e:
            msg = f"Pseudopotential file not found: {path_str}"
            raise ValueError(msg) from e
        except OSError as e:
            msg = f"Invalid pseudopotential path {path_str}: {e}"
            raise ValueError(msg) from e

        # Header check
        try:
            with resolved_path.open("rb") as f:
                header = f.read(100)
                try:
                    text_header = header.decode("utf-8")
                    if "<UPF" not in text_header and "PP_HEADER" not in text_header:
                        pass
                except UnicodeDecodeError as e:
                    msg = f"Pseudopotential file {path_str} does not appear to be a valid text-based UPF file."
                    raise ValueError(msg) from e
        except OSError as e:
            msg = f"Could not read pseudopotential file {path_str}: {e}"
            raise ValueError(msg) from e

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """
        Validates that pseudopotential files exist and are safe.
        """
        for elem, path_str in v.items():
            cls._validate_single_pp(elem, path_str)
        return v
