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
        default=DEFAULT_DFT_MIXING_BETA, gt=0.0, le=1.0, description="Initial mixing parameter for SCF"
    )
    smearing_type: str = Field(
        default=DEFAULT_DFT_SMEARING_TYPE, description="Type of smearing (e.g., 'mv', 'gaussian')"
    )
    smearing_width: PositiveFloat = Field(
        default=DEFAULT_DFT_SMEARING_WIDTH, description="Width of smearing in eV"
    )
    diagonalization: str = Field(
        default=DEFAULT_DFT_DIAGONALIZATION, description="Diagonalization algorithm"
    )

    # Strategy Multipliers
    # Note: mixing_beta_factor is used to REDUCE mixing_beta (new_beta = beta * factor)
    #       smearing_width_factor is used to INCREASE smearing_width (new_width = width * factor)
    mixing_beta_factor: float = Field(
        default=DEFAULT_DFT_MIXING_BETA_FACTOR,
        gt=0.0,
        le=1.0,
        description="Multiplier for mixing_beta reduction strategy",
    )
    smearing_width_factor: float = Field(
        default=DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
        gt=1.0,
        description="Multiplier for smearing_width increase strategy",
    )

    # Pseudopotentials
    # Parallelization
    n_workers: int = Field(
        default=1, gt=0, description="Number of parallel workers for DFT calculations"
    )

    # Pseudopotentials
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials_structure(cls, v: dict[str, str]) -> dict[str, str]:
        """Validates that keys are strings and values are non-empty strings."""
        for elem, path_str in v.items():
            if not isinstance(elem, str) or not elem:
                msg = "Pseudopotential keys must be non-empty element symbols"
                raise ValueError(msg)
            if not isinstance(path_str, str) or not path_str:
                msg = f"Pseudopotential path for {elem} must be a non-empty string"
                raise ValueError(msg)
        return v

    @staticmethod
    def _validate_single_path(elem: str, path_str: str) -> Path:
        """Helper to validate a single pseudopotential path."""
        import re

        MAX_PATH_LENGTH = 4096
        # Allow alphanumeric, dot, underscore, dash, slash, plus.
        SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_\-./\+]+$")

        if not path_str or not path_str.strip():
            msg = f"Pseudopotential path for {elem} cannot be empty"
            raise ValueError(msg)

        if len(path_str) > MAX_PATH_LENGTH:
            msg = f"Pseudopotential path for {elem} exceeds maximum length of {MAX_PATH_LENGTH}"
            raise ValueError(msg)

        if not SAFE_PATH_PATTERN.match(path_str):
            msg = f"Pseudopotential path for {elem} contains invalid characters. Allowed: alphanumeric, ., _, -, /, +"
            raise ValueError(msg)

        try:
            p = Path(path_str)
            # resolve(strict=True) will raise FileNotFoundError if file doesn't exist.
            resolved_path = p.resolve(strict=True)

            # Explicitly disallow symlinks
            if p.is_symlink():
                msg = f"Symlinks are not allowed for pseudopotentials: {path_str}"
                raise ValueError(msg)

            # Check if it's a file
            if not resolved_path.is_file():
                msg = f"Pseudopotential path is not a file: {resolved_path}"
                raise ValueError(msg)

        except FileNotFoundError as e:
            msg = f"Pseudopotential file not found: {path_str}"
            raise ValueError(msg) from e
        except OSError as e:
            msg = f"Invalid pseudopotential path {path_str}: {e}"
            raise ValueError(msg) from e

        return resolved_path

    @staticmethod
    def _validate_content(elem: str, path: Path) -> None:
        """Helper to validate pseudopotential content."""
        try:
            with path.open("rb") as f:
                # Read first 100 bytes
                header = f.read(100)
                # Check for typical UPF signatures or at least that it's not binary garbage
                try:
                    text_header = header.decode("utf-8")
                    if "<UPF" not in text_header and "PP_HEADER" not in text_header:
                        pass
                except UnicodeDecodeError as e:
                    msg = f"Pseudopotential file {path} for {elem} does not appear to be a valid text-based UPF file."
                    raise ValueError(msg) from e
        except OSError as e:
            msg = f"Could not read pseudopotential file {path}: {e}"
            raise ValueError(msg) from e

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """
        Validates that pseudopotential files exist.
        Allows absolute paths (e.g. system libraries).
        Disallows symlinks for security/portability.
        """
        for elem, path_str in v.items():
            resolved_path = cls._validate_single_path(elem, path_str)
            cls._validate_content(elem, resolved_path)

        return v
