from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, ValidationInfo, field_validator

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
    allowlist_paths: list[str] = Field(
        default_factory=list,
        description="List of allowed directory prefixes for pseudopotentials. If empty, restricts to project directory.",
    )
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str], info: ValidationInfo) -> dict[str, str]:
        """
        Validates that pseudopotential files exist and are within allowed paths.
        Allows absolute paths (e.g. system libraries) ONLY if they are in allowlist_paths.
        Disallows symlinks for security/portability.
        """
        # Get allowlist from model context if possible, or from the instance being validated
        # Field validation happens before model construction, so we access other fields via info.data
        allowed_prefixes = info.data.get("allowlist_paths", [])
        # Also always allow current working directory (project root)
        allowed_prefixes.append(str(Path.cwd().resolve()))

        for elem, path_str in v.items():
            cls._validate_single_path(elem, path_str, allowed_prefixes)

        return v

    @classmethod
    def _validate_single_path(cls, elem: str, path_str: str, allowed_prefixes: list[str]) -> None:
        if not path_str or not path_str.strip():
            msg = f"Pseudopotential path for {elem} cannot be empty"
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

            # Security: Check allowlist
            cls._check_allowlist(path_str, resolved_path, allowed_prefixes)

        except FileNotFoundError as e:
            msg = f"Pseudopotential file not found: {path_str}"
            raise ValueError(msg) from e
        except OSError as e:
            msg = f"Invalid pseudopotential path {path_str}: {e}"
            raise ValueError(msg) from e

        # Content Validation: Check for UPF header
        cls._validate_content(path_str, resolved_path)

    @staticmethod
    def _check_allowlist(path_str: str, resolved_path: Path, allowed_prefixes: list[str]) -> None:
        is_allowed = False
        for prefix in allowed_prefixes:
            try:
                allowed_path = Path(prefix).resolve()
                # check if resolved_path is relative to allowed_path
                if resolved_path.is_relative_to(allowed_path):
                    is_allowed = True
                    break
            except (ValueError, OSError):
                continue  # Invalid allowlist entry, skip

        if not is_allowed:
            msg = f"Pseudopotential {path_str} is not in allowed paths: {allowed_prefixes}"
            raise ValueError(msg)

    @staticmethod
    def _validate_content(path_str: str, resolved_path: Path) -> None:
        # We read the first few lines to ensure it looks like a pseudopotential file.
        # Standard UPF files start with <UPF version="..."> or similar XML/text.
        try:
            with resolved_path.open("rb") as f:
                # Read first 100 bytes
                header = f.read(100)
                # Check for typical UPF signatures or at least that it's not binary garbage
                # UPF v1/v2 are text-based.
                # We check for '<UPF' or 'PP_HEADER' (older formats) or just ensure it's text.
                try:
                    text_header = header.decode("utf-8")
                    if "<UPF" not in text_header and "PP_HEADER" not in text_header:
                         # Weak check, but better than nothing.
                         # If neither present, maybe warn? For now, we just enforce utf-8 decodeable.
                         pass
                except UnicodeDecodeError as e:
                     msg = f"Pseudopotential file {path_str} does not appear to be a valid text-based UPF file."
                     raise ValueError(msg) from e
        except OSError as e:
             msg = f"Could not read pseudopotential file {path_str}: {e}"
             raise ValueError(msg) from e
