from pathlib import Path
from types import TracebackType
from typing import Self


class DirectoryTransaction:
    """
    Context manager for transactional directory creation.
    Tracks created directories and rolls them back if an exception occurs.
    """

    def __init__(self) -> None:
        self.created_dirs: list[Path] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # An exception occurred, rollback created directories
            # Iterate in reverse order to delete children before parents
            for path in reversed(self.created_dirs):
                try:
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                except OSError:
                    # Log or ignore if we can't cleanup (e.g. permission lost)
                    # We can't log here easily without injecting logger,
                    # but typically we suppress rollback errors to propagate original error.
                    pass

    def create_directory(self, path: Path) -> None:
        """
        Creates a directory if it doesn't exist and tracks it for rollback.

        Args:
            path: Directory path to create.

        Raises:
            OSError: If directory creation fails.
        """
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(path)
        elif not path.is_dir():
            msg = f"Path exists but is not a directory: {path}"
            raise FileExistsError(msg)
