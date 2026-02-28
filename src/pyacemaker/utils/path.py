import os
import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


def _check_dangerous_chars(path: Path) -> None:
    s = str(path)
    if ".." in s:
        msg = f"Path traversal attempt detected (parent directory reference): {path}"
        raise ValueError(msg)
    if any(c in s for c in DANGEROUS_PATH_CHARS):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)

def _resolve_path(path: Path) -> Path:
    # Reject symlinks to prevent TOCTOU symlink attacks
    if path.is_symlink():
        msg = f"Path is a symlink, which is not allowed for security reasons: {path}"
        raise ValueError(msg)

    try:
        if path.exists():
            resolved = path.resolve(strict=True)
        elif path.parent.exists():
            resolved_parent = path.parent.resolve(strict=True)
            resolved = resolved_parent / path.name
        else:
            resolved = path.resolve(strict=False)
    except Exception as e:
        msg = f"Invalid path resolution: {path}"
        raise ValueError(msg) from e
    else:
        # Reject symlinks to prevent TOCTOU symlink attacks
        if path.is_symlink():
            msg = f"Path is a symlink, which is not allowed for security reasons: {path}"
            raise ValueError(msg)

        return resolved

def _check_allowed_roots(resolved: Path) -> None:
    base_dir = Path.cwd().resolve()
    allowed_roots = [
        base_dir,
        Path(tempfile.gettempdir()).resolve(),
        Path(DEFAULT_RAM_DISK_PATH).resolve(),
    ]

    for root in allowed_roots:
        try:
            common = Path(os.path.commonpath([root, resolved]))
            if common == root:
                return
        except ValueError:
            continue

    msg = f"Path traversal detected: {resolved} is outside allowed roots {allowed_roots}"
    raise ValueError(msg)

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
    _check_dangerous_chars(path)
    resolved = _resolve_path(path)
    _check_allowed_roots(resolved)
    return resolved
