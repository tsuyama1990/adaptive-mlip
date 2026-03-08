import os
import re
import tempfile
import urllib.parse
from pathlib import Path

from pyacemaker.domain_models.constants import DEFAULT_RAM_DISK_PATH

# Whitelist of allowed characters in paths: alphanumeric, space, dot, dash, underscore, and path separators.
ALLOWED_PATH_CHARS_REGEX = re.compile(r"^[\w\s.\-_/\\\:]+$")

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

    # Decode encoded characters (e.g. %2e%2e)
    unquoted = urllib.parse.unquote(s)

    # Regex whitelist validation
    if not ALLOWED_PATH_CHARS_REGEX.match(unquoted):
        msg = f"Path contains invalid characters: {path}"
        raise ValueError(msg)

    # Ensure filename doesn't start with dash (flag injection)
    if path.name.startswith("-"):
        msg = f"Filename cannot start with '-': {path.name}"
        raise ValueError(msg)

    try:
        # Canonicalize path using os.path.realpath to fully unfold symlinks safely.
        abs_path = os.path.abspath(unquoted)
        real_path_str = os.path.realpath(abs_path)
        resolved = Path(real_path_str)

        # If the file doesn't exist, we must check if its parent is valid.
        if not resolved.exists():
            # If the path parent doesn't exist, we must still enforce path restrictions on the string.
            pass

    except Exception as e:
         msg = f"Invalid path resolution: {path}"
         raise ValueError(msg) from e

    base_dir = Path.cwd().resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    ram_dir = Path(DEFAULT_RAM_DISK_PATH).resolve()

    allowed_roots = [
        str(os.path.realpath(str(base_dir))),
        str(os.path.realpath(str(temp_dir))),
        str(os.path.realpath(str(ram_dir)))
    ]

    is_safe = False

    for root in allowed_roots:
        # Use is_relative_to for robust path hierarchy check natively on the resolved path.
        # This accurately prevents symlink escapes and bounds checking.
        if resolved.is_relative_to(Path(root)):
            is_safe = True
            break

    if not is_safe:
         msg = f"Path traversal detected: {resolved} is outside allowed roots {allowed_roots}"
         raise ValueError(msg)

    return resolved
