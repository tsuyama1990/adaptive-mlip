import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


def validate_path_safe(path: Path) -> Path:  # noqa: C901
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

    try:
        # Canonicalize path.
        # Enforce strict=True if the path exists to catch symlink attacks immediately.
        # If it doesn't exist (e.g. output file), we must resolve based on parent.
        if path.exists():
            resolved = path.resolve(strict=True)
        # For non-existent files, check parent
        elif path.parent.exists():
            resolved_parent = path.parent.resolve(strict=True)
            # Combine resolved parent with filename
            resolved = resolved_parent / path.name
        else:
            # If even parent doesn't exist, this is likely unsafe or too deep.
            # Reject resolving completely to prevent TOCTOU and fake containment.
            pass

    except Exception as e:
        msg = f"Invalid path resolution: {path}"
        raise ValueError(msg) from e

    if not path.exists() and not path.parent.exists():
        msg = f"Parent directory does not exist: {path.parent}"
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
