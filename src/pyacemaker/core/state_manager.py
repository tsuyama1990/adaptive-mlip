from logging import Logger
from pathlib import Path
from typing import Any

from pyacemaker.core.loop import LoopState
from pyacemaker.domain_models.defaults import (
    LOG_STATE_LOAD_FAIL,
    LOG_STATE_LOAD_SUCCESS,
    LOG_STATE_SAVE_FAIL,
    LOG_STATE_SAVED,
)
from pyacemaker.utils.path import validate_path_safe


class StateManager:
    """
    Manages persistence of the active learning loop state.
    """

    def __init__(self, state_file: Path, logger: Logger, checkpoint_interval: int = 1) -> None:
        self.state_file = validate_path_safe(state_file)
        self.logger = logger
        self.state = LoopState()
        self.checkpoint_interval = checkpoint_interval
        self._last_saved_state_dump: dict[str, Any] | None = None

    def load(self) -> None:
        """Loads the iteration state."""
        try:
            self.state = LoopState.load(self.state_file)
            self._last_saved_state_dump = self.state.model_dump(mode="json")
            self.logger.info(LOG_STATE_LOAD_SUCCESS.format(iteration=self.state.iteration))
        except Exception as e:
            self.logger.warning(LOG_STATE_LOAD_FAIL.format(error=e))
            self.state = LoopState()
            self._last_saved_state_dump = None

    def save(self, force: bool = False) -> None:
        """
        Saves the current iteration state.

        Args:
            force: If True, bypasses dirty check and checkpoint interval (but still respects
                   duplicate save check if state hasn't changed, unless we want to force disk write).
                   Actually, dirty check logic: if state matches last saved dump, we skip.
        """
        try:
            current_dump = self.state.model_dump(mode="json")
            if current_dump == self._last_saved_state_dump:
                return

            # Throttling: Only save if interval met or forced
            # iteration 0 always saves (0 % N == 0).
            if not force and self.state.iteration > 0 and self.state.iteration % self.checkpoint_interval != 0:
                return

            self.state.save(self.state_file)
            self._last_saved_state_dump = current_dump
            self.logger.debug(LOG_STATE_SAVED.format(state=current_dump))
        except Exception as e:
            self.logger.warning(LOG_STATE_SAVE_FAIL.format(error=e))

    def rollback(self) -> None:
        """Restores the state from the last saved dump in memory."""
        if self._last_saved_state_dump:
            self.state = LoopState.model_validate(self._last_saved_state_dump)
            self.logger.info("Rolled back to last saved state.")
        else:
            self.logger.warning("No previous state to rollback to.")

    # Property wrappers removed to simplify indirection.
    # Callers should access self.state directly (e.g. self.state_manager.state.iteration)
