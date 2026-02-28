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
    Manages persistence of the active learning loop state safely to JSON configuration files.
    """

    def __init__(self, state_file: Path, logger: Logger, checkpoint_interval: int = 1) -> None:
        """
        Initializes StateManager and enforces security path checks on instantiation.

        Args:
            state_file: Destination relative or absolute Path target.
            logger: Runtime instantiated logging handle.
            checkpoint_interval: Controls periodic write interval throttle for `save`.
        """
        self.state_file = validate_path_safe(state_file)
        self.logger = logger
        self.state = LoopState()
        self.checkpoint_interval = checkpoint_interval
        self._last_saved_state_dump: dict[str, Any] | None = None

    def load(self) -> None:
        """
        Loads the iteration state safely overriding memory parameters with the contents
        of JSON persistent models. Handles missing or malformed states smoothly by returning
        defaults and issuing warnings.
        """
        try:
            safe_state_file = validate_path_safe(self.state_file)
            self.state = LoopState.load(safe_state_file)
            self._last_saved_state_dump = self.state.model_dump(mode="json")
            self.logger.info(LOG_STATE_LOAD_SUCCESS.format(iteration=self.state.iteration))
        except Exception as e:
            self.logger.warning(LOG_STATE_LOAD_FAIL.format(error=e))
            self.state = LoopState()
            self._last_saved_state_dump = None

    def save(self, force: bool = False) -> None:
        """
        Saves the current iteration state tracking object dynamically into storage via atomic operations.

        Args:
            force: Bypass standard dirty memory validation mapping and explicitly perform I/O write overrides.
        """
        try:
            current_dump = self.state.model_dump(mode="json")
            if current_dump == self._last_saved_state_dump:
                return

            if not force and self.state.iteration > 0 and self.state.iteration % self.checkpoint_interval != 0:
                return

            safe_state_file = validate_path_safe(self.state_file)
            self.state.save(safe_state_file)
            self._last_saved_state_dump = current_dump
            self.logger.debug(LOG_STATE_SAVED.format(state=current_dump))
        except Exception as e:
            self.logger.warning(LOG_STATE_SAVE_FAIL.format(error=e))

    def rollback(self) -> None:
        """
        Restores the state variable object from the last verified active serialized dict in the system caching memory block.
        Prevents full corruption from failed step execution processes inside distillation configurations.
        """
        if self._last_saved_state_dump:
            self.state = LoopState.model_validate(self._last_saved_state_dump)
            self.logger.info("Rolled back to last saved state.")
        else:
            self.logger.warning("No previous state to rollback to.")
