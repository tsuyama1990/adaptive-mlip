from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS


def resolve_path(path: str | Path) -> Path:
    """
    Resolves a path string or Path object to a resolved Path.
    Provides basic safety checks.
    """
    p = Path(path)
    return validate_path_safe(p)


def validate_path_safe(path: Path) -> Path:
    """
    Ensures path is safe using strict resolution and character allowlisting.
    Centralized utility for path validation.

    Args:
        path: The path to validate.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If the path contains dangerous characters or traversal attempts.
    """
    try:
        resolved = path.resolve()
    except Exception as e:
         msg = f"Invalid path resolution: {path}"
         raise ValueError(msg) from e

    s = str(resolved)

    # Check for dangerous patterns first
    if ".." in s:
         msg = f"Path traversal attempt detected: {path}"
         raise ValueError(msg)

    if any(c in s for c in DANGEROUS_PATH_CHARS):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)

    # Ensure filename doesn't start with dash (flag injection)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)

    return resolved
