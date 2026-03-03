import os
import tempfile
from pathlib import Path

from pyacemaker.domain_models.defaults import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


def _check_dangerous_chars(path: Path) -> None:
    s = str(path)
    if any(c in s for c in DANGEROUS_PATH_CHARS):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)


def _check_traversal(path: Path) -> None:
    s = str(path)
    if ".." in s:
        msg = f"Path traversal attempt detected (parent directory reference): {path}"
        raise ValueError(msg)


def _check_allowed_roots(resolved: Path) -> None:
    base_dir = Path.cwd().resolve()
    allowed_roots = [
        base_dir,
        Path(tempfile.gettempdir()).resolve(),
        Path(DEFAULT_RAM_DISK_PATH).resolve(),
    ]

    is_safe = False
    for root in allowed_roots:
        try:
            common = Path(os.path.commonpath([root, resolved]))
            if common == root:
                is_safe = True
                break
        except ValueError:
            continue

    if not is_safe:
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
                    symlinks, or resolves outside allowed directories (CWD, temp, /dev/shm).
    """
    _check_traversal(path)
    _check_dangerous_chars(path)

    # Block symlinks to prevent TOCTOU symlink traversal attacks
    if path.exists() and path.is_symlink():
        msg = f"Symlinks are forbidden for security reasons: {path}"
        raise ValueError(msg)

    try:
        # Canonicalize path strictly. Use os.path.realpath to fully resolve everything.
        # Strict resolution to block unpredictable traversal paths
        resolved_str = os.path.realpath(str(path))
        resolved = Path(resolved_str)

    except Exception as e:
        msg = f"Invalid path resolution: {path}"
        raise ValueError(msg) from e

    _check_allowed_roots(resolved)

    return resolved
