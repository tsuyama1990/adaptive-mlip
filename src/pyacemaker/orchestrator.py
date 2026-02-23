import json
from pathlib import Path

from pyacemaker.constants import (
    LOG_COMPUTED_PROPERTIES,
    LOG_GENERATED_CANDIDATES,
    LOG_INIT_MODULES,
    LOG_ITERATION_COMPLETED,
    LOG_MD_COMPLETED,
    LOG_MODULE_INIT_FAIL,
    LOG_MODULES_INIT_SUCCESS,
    LOG_POTENTIAL_TRAINED,
    LOG_PROJECT_INIT,
    LOG_START_ITERATION,
    LOG_START_LOOP,
    LOG_STATE_LOAD_FAIL,
    LOG_STATE_LOAD_SUCCESS,
    LOG_STATE_SAVE_FAIL,
    LOG_STATE_SAVED,
    LOG_WORKFLOW_COMPLETED,
    LOG_WORKFLOW_CRASHED,
)
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.logger import setup_logger


class Orchestrator:
    """
    Central controller for the PYACEMAKER workflow.
    Manages the lifecycle of the active learning loop, error handling, and state persistence.
    """

    def __init__(self, config: PyAceConfig) -> None:
        """
        Initializes the Orchestrator with a configuration.

        Args:
            config: Validated PyAceConfig object.
        """
        self.config = config
        self.logger = setup_logger(config=config.logging, project_name=config.project_name)
        self.iteration = 0
        self.state_file = Path(config.workflow.state_file_path)

        # Core modules (placeholders for Cycle 01)
        self.generator: BaseGenerator | None = None
        self.oracle: BaseOracle | None = None
        self.trainer: BaseTrainer | None = None
        self.engine: BaseEngine | None = None

        self.load_state()
        self.logger.info(LOG_PROJECT_INIT.format(project_name=config.project_name))

    def initialize_modules(self) -> None:
        """
        Initializes the core modules (Generator, Oracle, Trainer, Engine).

        Raises:
            RuntimeError: If module initialization fails.
        """
        self.logger.info(LOG_INIT_MODULES)
        try:
            # In future cycles, we will instantiate concrete classes here based on config.
            # For Cycle 01, we just set placeholders if they aren't injected (for testing).
            # This method acts as a composition root.
            pass

        except Exception as e:
            self.logger.exception("Failed to initialize modules")
            raise RuntimeError(LOG_MODULE_INIT_FAIL.format(error=e)) from e

        self.logger.info(LOG_MODULES_INIT_SUCCESS)

    def save_state(self) -> None:
        """Saves the current iteration state to a JSON file."""
        state = {"iteration": self.iteration}
        try:
            with self.state_file.open("w") as f:
                json.dump(state, f)
            self.logger.debug(LOG_STATE_SAVED.format(state=state))
        except Exception as e:
            self.logger.warning(LOG_STATE_SAVE_FAIL.format(error=e))

    def load_state(self) -> None:
        """Loads the iteration state from a JSON file if it exists."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r") as f:
                    state = json.load(f)
                    self.iteration = state.get("iteration", 0)
                self.logger.info(LOG_STATE_LOAD_SUCCESS.format(iteration=self.iteration))
            except Exception as e:
                self.logger.warning(LOG_STATE_LOAD_FAIL.format(error=e))

    def _run_active_learning_step(self) -> None:
        """Executes a single step of the active learning loop."""

        # 1. Generate & Label Candidates (Streaming)
        total_candidates = 0
        n_candidates = self.config.workflow.n_candidates
        batch_size = self.config.workflow.batch_size

        if self.generator and self.oracle:
            # Generator returns iterator
            candidate_stream = self.generator.generate(n_candidates=n_candidates)

            # Oracle consumes iterator and returns iterator
            # Note: Depending on Oracle implementation, it might batch internally or we handle it here.
            # To strictly follow ABC, pass iterator.
            labelled_stream = self.oracle.compute(candidate_stream, batch_size=batch_size)

            for _ in labelled_stream:
                total_candidates += 1
                # TODO: Accumulate training data or stream to trainer

            self.logger.info(LOG_COMPUTED_PROPERTIES.format(count=total_candidates))

        elif self.generator:
            # Just consume generator if no oracle (mock mode)
            for _ in self.generator.generate(n_candidates=n_candidates):
                total_candidates += 1
            self.logger.info(LOG_GENERATED_CANDIDATES.format(count=total_candidates))

        # 3. Train potential
        if self.trainer:
            # Note: Training usually requires full dataset, but we can't load all history into memory.
            # Trainer implementation will handle file-based datasets.
            _ = self.trainer.train(training_data=[])
            self.logger.info(LOG_POTENTIAL_TRAINED)

        # 4. Run MD
        if self.engine:
            self.engine.run(structure=None, potential=None)  # type: ignore[arg-type]
            self.logger.info(LOG_MD_COMPLETED)

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        self.logger.info(LOG_START_LOOP)

        try:
            self.initialize_modules()

            max_iterations = self.config.workflow.max_iterations
            start_iter = self.iteration
            checkpoint_interval = self.config.workflow.checkpoint_interval

            for i in range(start_iter, max_iterations):
                self.iteration = i + 1
                self.logger.info(
                    LOG_START_ITERATION.format(
                        iteration=self.iteration, max_iterations=max_iterations
                    )
                )

                self._run_active_learning_step()

                # Checkpoint based on interval
                if self.iteration % checkpoint_interval == 0:
                    self.save_state()

                self.logger.info(LOG_ITERATION_COMPLETED.format(iteration=self.iteration))

        except Exception as e:
            self.logger.critical(LOG_WORKFLOW_CRASHED.format(error=e))
            raise

        self.logger.info(LOG_WORKFLOW_COMPLETED)
