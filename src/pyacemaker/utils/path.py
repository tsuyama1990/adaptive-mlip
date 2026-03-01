import os
import tempfile
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS, DEFAULT_RAM_DISK_PATH


class PathValidator:
    """Orchestrates secure path validation routines."""

    def __init__(self, allowed_roots: list[Path] | None = None) -> None:
        self.allowed_roots = allowed_roots

    def validate(self, path: Path) -> Path:
        self._validate_path_characters(path)
        self._validate_path_symlinks(path)
        resolved = self._resolve_path(path)
        self._validate_path_containment(resolved)
        return resolved

    def _validate_path_characters(self, path: Path) -> None:
        s = str(path)
        if any(c in s for c in DANGEROUS_PATH_CHARS):
            raise ValueError(f"Path contains invalid characters: {path}")
        if path.name.startswith("-"):
            raise ValueError(f"Filename cannot start with '-': {path.name}")

    def _validate_path_symlinks(self, path: Path) -> None:
        # Prevent symlink traversal attacks prior to resolution.
        # We must use lstat to check if it's a symlink directly, even if it points to nowhere.
        try:
            # os.lstat raises FileNotFoundError if it doesn't exist, which is fine
            # If it exists (even as a broken symlink), it will succeed.
            if os.path.islink(str(path)):
                raise ValueError(f"Symlinks are not permitted for security reasons: {path}")

            # For added safety against directory symlinks in the path itself:
            # Check the resolved parent is not traversing a symlink boundary unexpectedly
            # if we wanted strict containment, but islink handles the terminal file.
        except FileNotFoundError:
            # If the file does not exist, it cannot be a symlink yet.
            # But what if its parent is a symlink? We enforce strict=True on parent resolution.
            pass
        except OSError as e:
            raise ValueError(f"Failed to stat path for symlinks: {path}") from e

    def _resolve_path(self, path: Path) -> Path:
        try:
            if path.exists():
                return path.resolve(strict=True)
            else:
                resolved_parent = path.parent.resolve(strict=True)
                return resolved_parent / path.name
        except Exception as e:
            raise ValueError(f"Invalid path resolution: {path}") from e

    def _validate_path_containment(self, resolved: Path) -> None:
        allowed = self.allowed_roots
        if allowed is None:
            base_dir = Path.cwd().resolve()
            env_roots_raw = os.environ.get("PYACEMAKER_ALLOWED_ROOTS", "")
            if env_roots_raw:
                 allowed = [Path(os.path.realpath(p)) for p in env_roots_raw.split(":") if p]
            else:
                 allowed = [
                     base_dir,
                     Path(tempfile.gettempdir()).resolve(),
                     Path(DEFAULT_RAM_DISK_PATH).resolve()
                 ]

        is_safe = False
        for root in allowed:
            try:
                root_real = os.path.realpath(str(root))
                if os.path.commonpath([str(resolved), root_real]) == root_real:
                    is_safe = True
                    break
            except ValueError:
                continue

        if not is_safe:
             raise ValueError(f"Path traversal detected: {resolved} is outside allowed roots {allowed}")


def validate_path_safe(path: Path, allowed_roots: list[Path] | None = None) -> Path:
    """Legacy wrapper delegating to PathValidator."""
    validator = PathValidator(allowed_roots)
    return validator.validate(path)
