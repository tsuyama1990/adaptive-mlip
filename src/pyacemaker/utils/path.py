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
    # Pre-check original path for raw directory traversal string before any resolution
    if ".." in str(path):
        msg = f"Path traversal attempt detected (parent directory reference): {path}"
        raise ValueError(msg)

    try:
        import os

        is_link = False
        try:
            is_link = path.is_symlink()
        except OSError:
            pass

        # Canonicalize path using realpath which aggressively resolves all symlinks
        # without raising errors on non-existent files, avoiding TOCTOU bugs.
        resolved = Path(os.path.realpath(path))

        # Explicitly check resolved path for traversal components per security audit
        if ".." in str(resolved):
             msg = f"Resolved path still contains traversal components: {resolved}"
             raise ValueError(msg)

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
    import os
    for root in allowed_roots:
        # Robust check using os.path.realpath and os.path.commonpath to ensure resolved path is under root securely
        # This addresses the audit finding specifically requiring commonpath over is_relative_to
        try:
            r_str = str(root)
            res_str = str(resolved)
            common = os.path.commonpath([r_str, res_str])
            if common == r_str:
                is_safe = True
                break
        except ValueError:
            continue

    if not is_safe:
         msg = f"Path traversal detected: {resolved} is outside allowed roots {allowed_roots}"
         raise ValueError(msg)

    return resolved
