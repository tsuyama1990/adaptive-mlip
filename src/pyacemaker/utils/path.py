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
            # Use pathlib.is_relative_to for robust, secure containment checks
            if resolved.is_relative_to(root):
                is_safe = True
                break
        except Exception:
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

    try:
        # Canonicalize path strictly. Use os.path.realpath to fully resolve everything.
        # TOCTOU Prevention: We resolve the real path first, to catch any symlinks pointing
        # out of bounds. Even if a symlink exists within a safe directory, its realpath
        # will point elsewhere and get correctly trapped by `_check_allowed_roots`.
        resolved_str = os.path.realpath(str(path))
        resolved = Path(resolved_str)

        # Additionally, verify the ORIGINAL path string (even if a symlink) hasn't traversed roots
        # This double check ensures absolute symlink chains must stem from authorized roots.
        original_base = Path(os.path.abspath(str(path)))
        _check_allowed_roots(original_base)

    except Exception as e:
        msg = f"Invalid path resolution: {path}"
        raise ValueError(msg) from e

    _check_allowed_roots(resolved)

    return resolved
