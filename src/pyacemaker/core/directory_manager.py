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

        try:
            # Batch creation: Iterate and create.
            # Use exist_ok=True to handle re-runs and parallelism safely.
            # Implicitly checks permissions via OS exceptions.
            for p in paths.values():
                p.mkdir(parents=True, exist_ok=True)

        except OSError as e:
            msg = f"Failed to setup directory structure for iteration {iteration}: {e}"
            self.logger.critical(msg)
            raise OrchestratorError(msg) from e

        return paths
