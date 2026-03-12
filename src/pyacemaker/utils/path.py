import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


# ruff: noqa: C901
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
    s = str(path)

    # Check for dangerous patterns in string representation BEFORE resolve
    if ".." in s:
        msg = f"Path traversal attempt detected (parent directory reference): {path}"
        raise ValueError(msg)

    # Check absolute path prefixes to ensure we don't start with dangerous roots not in allowed directly
    tmp_path_str = tempfile.gettempdir()
    shm_path_str = DEFAULT_RAM_DISK_PATH
    if path.is_absolute() and not str(path).startswith((tmp_path_str, shm_path_str, str(Path.cwd().resolve()))):
        msg = f"Path attempts to access external root: {path}"
        raise ValueError(msg)

    if any(c in s for c in DANGEROUS_PATH_CHARS):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)

    # Ensure filename doesn't start with dash (flag injection)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)

    # Prevent symlink traversal attacks BEFORE resolution
    if path.is_symlink():
        msg = f"Symlink path traversal attacks detected: {path}"
        raise ValueError(msg)

    import os
    try:
        # Use realpath to resolve all symlinks safely
        real_path_str = os.path.realpath(str(path))
        resolved = Path(real_path_str)

        # Verify parent exists if file doesn't
        if not resolved.exists() and not resolved.parent.exists():
            _raise_parent_error(path)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        msg = f"Invalid path resolution: {path}"
        raise ValueError(msg) from e

    base_dir = Path.cwd().resolve()

    # Allowed roots: CWD, System Temp, RAM Disk
    allowed_roots = [
        base_dir,
        Path(tempfile.gettempdir()).resolve(),
        Path(DEFAULT_RAM_DISK_PATH).resolve(),
    ]

    is_safe = False
    for root in allowed_roots:
        if resolved.is_relative_to(root):
            is_safe = True
            break

    if not is_safe:
        msg = f"Path traversal detected: {resolved} is outside allowed roots {allowed_roots}"
        raise ValueError(msg)

    return resolved

def _raise_parent_error(path: Path) -> None:
    msg = f"Parent directory does not exist for resolution: {path}"
    raise ValueError(msg)
