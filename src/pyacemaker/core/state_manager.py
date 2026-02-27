from logging import Logger
from pathlib import Path

from pyacemaker.core.loop import LoopState
from pyacemaker.domain_models.workflow import WorkflowStep
from pyacemaker.domain_models.defaults import (
    LOG_STATE_LOAD_FAIL,
    LOG_STATE_LOAD_SUCCESS,
    LOG_STATE_SAVE_FAIL,
    LOG_STATE_SAVED,
)


class StateManager:
    """
    Manages persistence of the active learning loop state.
    """

    def __init__(self, state_file: Path, logger: Logger) -> None:
        self.state_file = state_file
        self.logger = logger
        self.state = LoopState()

    def load(self) -> None:
        """Loads the iteration state."""
        try:
            self.state = LoopState.load(self.state_file)
            self.logger.info(LOG_STATE_LOAD_SUCCESS.format(iteration=self.state.iteration))
        except Exception as e:
            self.logger.warning(LOG_STATE_LOAD_FAIL.format(error=e))
            self.state = LoopState()

    def save(self) -> None:
        """Saves the current iteration state."""
        try:
            self.state.save(self.state_file)
            self.logger.debug(LOG_STATE_SAVED.format(state=self.state.model_dump()))
        except Exception as e:
            self.logger.warning(LOG_STATE_SAVE_FAIL.format(error=e))

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

    @property
    def current_step(self) -> WorkflowStep | None:
        return self.state.current_step

    @current_step.setter
    def current_step(self, value: WorkflowStep | None) -> None:
        self.state.current_step = value

    @property
    def mode(self) -> str:
        return self.state.mode

    @mode.setter
    def mode(self, value: str) -> None:
        self.state.mode = value
