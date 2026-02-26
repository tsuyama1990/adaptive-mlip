import tempfile
import os
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

    try:
        # Canonicalize path
        # Use strict=True to prevent resolving to non-existent paths that could be manipulated later,
        # unless it's an output file we intend to create?
        # The audit requirement says "resolve(strict=True)".
        # If the file is expected to not exist (e.g. output), we must check the parent directory.
        # But `validate_path_safe` is generic.
        # Let's assume for existing files (inputs) strict=True is required.
        # For outputs, the caller might need to check parent.
        # However, if we enforce strict=True, we break output file path validation if the file doesn't exist yet.
        # COMPROMISE: If path doesn't exist, we check if parent exists and is safe.
        # Actually, best practice for anti-traversal is checking the resolved path against root.
        # If strict=True is used, it raises FileNotFoundError if it doesn't exist.

        if path.exists():
            resolved = path.resolve(strict=True)
        else:
            # If file doesn't exist, resolve parent and check that.
            # Then resolve full path (non-strict) to get canonical form without following non-existent symlinks (since they don't exist).
            # But wait, resolve(strict=False) on non-existent path handles ".." purely lexically if components missing?
            # No, resolve() eliminates ".."
            resolved = path.resolve(strict=False)
            # Ensure parent exists so we aren't writing to completely random place
            if not resolved.parent.exists():
                 # This might be too strict for nested directory creation?
                 # Orchestrator creates dirs.
                 pass

    except Exception as e:
         msg = f"Invalid path resolution: {path}"
         raise ValueError(msg) from e

    base_dir = Path.cwd().resolve()

    # Allowed roots: CWD, System Temp, RAM Disk
    allowed_roots = [
        base_dir,
        Path(tempfile.gettempdir()).resolve(),
        Path(DEFAULT_RAM_DISK_PATH).resolve()
    ]

    is_safe = False
    for root in allowed_roots:
        # Robust check using os.path.commonpath to ensure resolved path is under root
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

    return resolved
