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
from pyacemaker.utils.misc import batched


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
        self.state_file = Path(f"{config.project_name}_state.json")

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
            pass

        except Exception as e:
            # LOG_MODULE_INIT_FAIL has a placeholder {error} which we use in the RuntimeError message
            # For logging exception, we just say "Failed to initialize modules"
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

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        self.logger.info(LOG_START_LOOP)

        try:
            self.initialize_modules()

            max_iterations = self.config.workflow.max_iterations
            start_iter = self.iteration

            for i in range(start_iter, max_iterations):
                self.iteration = i + 1
                self.logger.info(LOG_START_ITERATION.format(iteration=self.iteration, max_iterations=max_iterations))

                # Active Learning Loop Logic

                # 1. Generate & Label Candidates (Streaming/Batching)
                total_candidates = 0
                if self.generator and self.oracle:
                    # Generator returns iterator
                    candidate_stream = self.generator.generate(n_candidates=10)

                    # Process in batches
                    for batch in batched(candidate_stream, n=5):
                        # Convert tuple batch to list for Oracle
                        structures = list(batch)
                        results = self.oracle.compute(structures=structures)
                        total_candidates += len(results)
                        # TODO: Accumulate training data or stream to trainer

                    self.logger.info(LOG_COMPUTED_PROPERTIES.format(count=total_candidates))

                elif self.generator:
                     # Just consume generator if no oracle (mock mode)
                     for _ in self.generator.generate(n_candidates=10):
                         total_candidates += 1
                     self.logger.info(LOG_GENERATED_CANDIDATES.format(count=total_candidates))

                # 3. Train potential
                if self.trainer:
                    _ = self.trainer.train(training_data=[])
                    self.logger.info(LOG_POTENTIAL_TRAINED)

                # 4. Run MD
                if self.engine:
                    self.engine.run(structure=None, potential=None) # type: ignore[arg-type]
                    self.logger.info(LOG_MD_COMPLETED)

                # Checkpoint
                self.save_state()
                self.logger.info(LOG_ITERATION_COMPLETED.format(iteration=self.iteration))

        except Exception as e:
            self.logger.critical(LOG_WORKFLOW_CRASHED.format(error=e))
            raise

        self.logger.info(LOG_WORKFLOW_COMPLETED)
