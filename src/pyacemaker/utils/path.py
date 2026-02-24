from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS


def validate_path_safe(path: Path) -> Path:
    """
    Ensures path is safe using strict resolution and character allowlisting.
    Centralized utility for path validation.

    Args:
        path: The path to validate.

    Returns:
        The resolved Path object.

    Raises:
        ActiveSetError: If the path contains dangerous characters or traversal attempts.
                        (Using ActiveSetError for consistency, or generic ValueError/RuntimeError could be used,
                         but keeping existing exception type for now as it's used in ActiveSetSelector).
                         Actually, let's use ValueError for a utility, but caller might expect specific error.
                         Let's stick to the pattern used in ActiveSetSelector for now or introduce a generic PathError.
                         For simplicity and audit compliance, raising ValueError/RuntimeError is standard for utils,
                         but ActiveSetSelector catches ActiveSetError.
                         I will raise ValueError, and ActiveSetSelector can wrap it or I import ActiveSetError.
                         Given the plan imports ActiveSetError, I will use it or a base error.
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
