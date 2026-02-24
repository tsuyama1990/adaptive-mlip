import os
from logging import Logger
from pathlib import Path

from pyacemaker.core.exceptions import OrchestratorError
from pyacemaker.domain_models.defaults import TEMPLATE_ITER_DIR


class DirectoryManager:
    """
    Manages directory structures for the active learning workflow.
    """

    def __init__(self, base_dir: Path, logger: Logger) -> None:
        self.base_dir = base_dir
        self.logger = logger

    def _ensure_directory(self, path: Path) -> None:
        """
        Creates a directory and verifies write permissions.
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                msg = f"Path {path} exists but is not a directory."
                raise RuntimeError(msg)
            if not os.access(path, os.W_OK):
                msg = f"Directory {path} is not writable."
                raise PermissionError(msg)
        except OSError as e:
            self.logger.critical(f"Failed to create directory {path}: {e}")
            raise

    def setup_iteration(self, iteration: int) -> dict[str, Path]:
        """
        Creates the directory structure for the current iteration.
        """
        iter_dirname = TEMPLATE_ITER_DIR.format(iteration=iteration)
        iter_dir = self.base_dir / iter_dirname

        paths = {
            "root": iter_dir,
            "candidates": iter_dir / "candidates",
            "dft_calc": iter_dir / "dft_calc",
            "training": iter_dir / "training",
            "md_run": iter_dir / "md_run",
        }

        created_paths: list[Path] = []
        try:
            for p in paths.values():
                if not p.exists():
                    self._ensure_directory(p)
                    created_paths.append(p)
                else:
                    self._ensure_directory(p)
        except Exception as e:
            self.logger.exception("Failed to create iteration directories")
            # Attempt rollback
            for p in reversed(created_paths):
                try:
                    if p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                except OSError:
                    self.logger.warning(f"Failed to rollback directory creation for {p}")

            msg = f"Failed to setup directory: {e}"
            raise OrchestratorError(msg) from e

        return paths
