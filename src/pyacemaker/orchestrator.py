import json
from pathlib import Path

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
        self.state_file = Path(f"{config.project_name}_state.json")

        # Core modules (placeholders for Cycle 01)
        self.generator: BaseGenerator | None = None
        self.oracle: BaseOracle | None = None
        self.trainer: BaseTrainer | None = None
        self.engine: BaseEngine | None = None

        self.load_state()
        self.logger.info(f"Project: {config.project_name} initialized.")

    def initialize_modules(self) -> None:
        """
        Initializes the core modules (Generator, Oracle, Trainer, Engine).

        Raises:
            RuntimeError: If module initialization fails.
        """
        self.logger.info("Initializing modules...")
        try:
            # In future cycles, we will instantiate concrete classes here based on config.
            pass

        except Exception as e:
            msg = f"Module initialization failed: {e}"
            self.logger.exception("Failed to initialize modules")
            raise RuntimeError(msg) from e

        self.logger.info("Modules initialized (Mock mode for Cycle 01).")

    def save_state(self) -> None:
        """Saves the current iteration state to a JSON file."""
        state = {"iteration": self.iteration}
        try:
            with self.state_file.open("w") as f:
                json.dump(state, f)
            self.logger.debug(f"State saved: {state}")
        except Exception as e:
            self.logger.warning(f"Failed to save state: {e}")

    def load_state(self) -> None:
        """Loads the iteration state from a JSON file if it exists."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r") as f:
                    state = json.load(f)
                    self.iteration = state.get("iteration", 0)
                self.logger.info(f"Resuming from iteration {self.iteration}")
            except Exception as e:
                self.logger.warning(f"Failed to load state, starting fresh: {e}")

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        self.logger.info("Starting active learning loop.")

        try:
            self.initialize_modules()

            max_iterations = self.config.workflow.max_iterations
            start_iter = self.iteration

            for i in range(start_iter, max_iterations):
                self.iteration = i + 1
                self.logger.info(f"Starting Iteration {self.iteration}/{max_iterations}")

                # Active Learning Loop Logic

                # 1. Generate candidates
                if self.generator:
                    candidates = list(self.generator.generate(n_candidates=10)) # Consume iterator
                    self.logger.info(f"Generated {len(candidates)} candidates")

                # 2. Compute DFT
                if self.oracle:
                    results = self.oracle.compute(structures=[], batch_size=5)
                    self.logger.info(f"Computed properties for {len(results)} structures")

                # 3. Train potential
                if self.trainer:
                    _ = self.trainer.train(training_data=[])
                    self.logger.info("Potential trained")

                # 4. Run MD
                if self.engine:
                    self.engine.run(structure=None, potential=None) # type: ignore[arg-type]
                    self.logger.info("MD simulation completed")

                # Checkpoint
                self.save_state()
                self.logger.info(f"Iteration {self.iteration} completed.")

        except Exception as e:
            self.logger.critical(f"Workflow crashed: {e}")
            raise

        self.logger.info("Workflow completed.")
