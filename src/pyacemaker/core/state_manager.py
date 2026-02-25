import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from pathlib import Path

from pyacemaker.core.loop import LoopState
from pyacemaker.domain_models.defaults import (
    LOG_STATE_LOAD_FAIL,
    LOG_STATE_LOAD_SUCCESS,
    LOG_STATE_SAVE_FAIL,
    LOG_STATE_SAVED,
)


class StateManager:
    """
    Manages persistence of the active learning loop state.
    Uses asynchronous I/O and atomic writes for performance and data integrity.
    """

    def __init__(self, state_file: Path, logger: Logger) -> None:
        self.state_file = state_file
        self.logger = logger
        self.state = LoopState()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="StateManager")

    def load(self) -> None:
        """Loads the iteration state synchronously (required for startup)."""
        try:
            self.state = LoopState.load(self.state_file)
            self.logger.info(LOG_STATE_LOAD_SUCCESS.format(iteration=self.state.iteration))
        except Exception as e:
            self.logger.warning(LOG_STATE_LOAD_FAIL.format(error=e))
            self.state = LoopState()

    def save(self) -> None:
        """Saves the current iteration state asynchronously."""
        # Create a copy of the state model to ensure thread safety during serialization
        # Use mode='json' to ensure all types (like Path) are serialized to JSON-compatible format
        state_dump = self.state.model_dump(mode="json")
        self._executor.submit(self._save_task, state_dump, self.state_file)

    def _save_task(self, state_data: dict, filepath: Path) -> None:
        """Background task for atomic save."""
        try:
            # Atomic write: write to temp file then move
            # Use directory of target file to ensure same filesystem for atomic rename
            parent_dir = filepath.parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(mode="w", dir=parent_dir, delete=False) as tmp_file:
                json.dump(state_data, tmp_file, indent=2)
                tmp_path = Path(tmp_file.name)

            # Atomic rename (POSIX) / Replace (Windows with recent Python)
            shutil.move(str(tmp_path), str(filepath))

            self.logger.debug(LOG_STATE_SAVED.format(state=state_data))
        except Exception as e:
            self.logger.warning(LOG_STATE_SAVE_FAIL.format(error=e))
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and tmp_path.exists(): # type: ignore
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def shutdown(self) -> None:
        """Ensures all pending saves are completed."""
        self._executor.shutdown(wait=True)

    @property
    def iteration(self) -> int:
        return self.state.iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self.state.iteration = value

    @property
    def current_potential(self) -> Path | None:
        return self.state.current_potential

    @current_potential.setter
    def current_potential(self, value: Path | None) -> None:
        self.state.current_potential = value
