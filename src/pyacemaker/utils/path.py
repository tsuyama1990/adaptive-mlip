import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DEFAULT_RAM_DISK_PATH, MALICIOUS_SHELL_PATTERN


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
    import os
    import urllib.parse

    # Decode to catch encoded traversals like %2E%2E
    s = urllib.parse.unquote(str(path))

    # Check for dangerous patterns in string representation BEFORE resolve
    if ".." in s or "\\.." in s or "/.." in s:
        msg = f"Path traversal attempt detected (parent directory reference): {path}"
        raise ValueError(msg)

    # Resolve first to avoid TOCTOU on existence/symlink checks.
    # realpath resolves symlinks.
    try:
        resolved_str = os.path.realpath(path)
        resolved = Path(resolved_str)
    except Exception as e:
        msg = f"Invalid path resolution: {path}"
        raise ValueError(msg) from e

    # Verify it resolves correctly
    if not resolved.exists() and not resolved.parent.exists():
        msg = f"Parent directory does not exist: {resolved.parent}"
        raise ValueError(msg)

    # Re-check the fully resolved string for bad patterns
    resolved_s = urllib.parse.unquote(str(resolved))
    if ".." in resolved_s or "\\.." in resolved_s or "/.." in resolved_s:
        msg = f"Path traversal attempt detected in resolved path: {resolved}"
        raise ValueError(msg)

    import re
    if re.search(MALICIOUS_SHELL_PATTERN, s):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)

    # Ensure filename doesn't start with dash (flag injection)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)

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
