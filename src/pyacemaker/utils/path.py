import os
import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


def validate_path_safe(path: Path, allowed_roots: list[Path] | None = None) -> Path:
    """
    Ensures path is safe using strict resolution and character allowlisting.
    Centralized utility for path validation.

    Args:
        path: The path to validate.
        allowed_roots: Optional list of allowed root directories. Defaults to env/cwd/temp.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If the path contains dangerous characters, traversal attempts,
                    or resolves outside allowed directories.
    """
    s = str(path)

    if any(c in s for c in DANGEROUS_PATH_CHARS):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)

    # Ensure filename doesn't start with dash (flag injection)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)

    try:
        # Canonicalize path atomically using strict resolution if it's meant to exist.
        # This prevents symlink substitution race conditions (TOCTOU).
        # We always resolve the absolute path and require the parent directory to exist at minimum.
        if path.exists():
            resolved = path.resolve(strict=True)
        else:
            resolved_parent = path.parent.resolve(strict=True)
            resolved = resolved_parent / path.name
    except Exception as e:
         msg = f"Invalid path resolution: {path}"
         raise ValueError(msg) from e

    if allowed_roots is None:
        base_dir = Path.cwd().resolve()

        # Pull allowed roots from env var if exists
        env_roots_raw = os.environ.get("PYACEMAKER_ALLOWED_ROOTS", "")
        if env_roots_raw:
             allowed_roots = [Path(os.path.realpath(p)) for p in env_roots_raw.split(":") if p]
        else:
             allowed_roots = [
                 base_dir,
                 Path(tempfile.gettempdir()).resolve(),
                 Path(DEFAULT_RAM_DISK_PATH).resolve()
             ]

    is_safe = False
    for root in allowed_roots:
        try:
            root_real = os.path.realpath(str(root))
            if os.path.commonpath([str(resolved), root_real]) == root_real:
                is_safe = True
                break
        except ValueError:
            continue

    if not is_safe:
         msg = f"Path traversal detected: {resolved} is outside allowed roots {allowed_roots}"
         raise ValueError(msg)

    return resolved
