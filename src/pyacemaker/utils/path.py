import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


def validate_path_safe(path: Path) -> Path:
    """
    Ensures path is safe using strict resolution and character allowlisting.
    Centralized utility for path validation.

    Args:
        path: The path to validate.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If the path contains dangerous characters, traversal attempts,
                    or resolves outside allowed directories (CWD, temp, /dev/shm).
    """
    try:
        import os
        # Canonicalize path using realpath which aggressively resolves all symlinks
        # without raising errors on non-existent files, avoiding TOCTOU bugs.
        resolved = Path(os.path.realpath(path))
    except Exception as e:
         msg = f"Invalid path resolution: {path}"
         raise ValueError(msg) from e

    s = str(resolved)

    if any(c in s for c in DANGEROUS_PATH_CHARS):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)

    # Ensure filename doesn't start with dash (flag injection)
    if resolved.name.startswith("-"):
        msg = f"Filename cannot start with '-': {resolved.name}"
        raise ValueError(msg)

    base_dir = Path.cwd().resolve()

    # Allowed roots: CWD, System Temp, RAM Disk
    allowed_roots = [
        base_dir,
        Path(tempfile.gettempdir()).resolve(),
        Path(DEFAULT_RAM_DISK_PATH).resolve()
    ]

    is_safe = False
    for root in allowed_roots:
        # Robust check using is_relative_to to ensure resolved path is under root securely
        if resolved.is_relative_to(root):
            is_safe = True
            break

    if not is_safe:
         msg = f"Path traversal detected: {resolved} is outside allowed roots {allowed_roots}"
         raise ValueError(msg)

    return resolved
