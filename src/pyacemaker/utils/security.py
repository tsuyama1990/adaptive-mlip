from pathlib import Path


def validate_path_containment(target_path: str | Path, allowed_base_dir: str | Path) -> Path:
    """
    Strictly verifies that the target path falls inside the accepted allowed base directory.
    This prevents directory traversal attacks (e.g., passing "../../../etc/passwd").

    Args:
        target_path: The path to validate.
        allowed_base_dir: The directory that must contain the target path.

    Returns:
        The canonical, absolute path to the target.

    Raises:
        FileNotFoundError: If the target path does not exist.
        ValueError: If the target path is not a file or is not contained within allowed_base_dir.
    """
    target_p = Path(target_path)

    if target_p.is_symlink():
        msg = f"Symlinks are not allowed for security reasons: {target_path}"
        raise ValueError(msg)

    try:
        canonical_path = target_p.resolve(strict=True)
    except FileNotFoundError as e:
        msg = f"Path does not exist: {target_path}"
        raise FileNotFoundError(msg) from e

    canonical_allowed_dir = Path(allowed_base_dir).resolve(strict=True)

    if not canonical_path.is_relative_to(canonical_allowed_dir):
        msg = f"Path {canonical_path} is outside allowed directory {canonical_allowed_dir}"
        raise ValueError(msg)

    if not canonical_path.is_file():
        msg = f"Path must be a file: {canonical_path}"
        raise ValueError(msg)

    return canonical_path
