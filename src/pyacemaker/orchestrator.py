import contextlib
import json
import os
import shutil
from pathlib import Path

from ase.io import iread, write

from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.defaults import (
    FILENAME_CANDIDATES,
    FILENAME_TRAINING,
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
    TEMPLATE_ITER_DIR,
    TEMPLATE_POTENTIAL_FILE,
)
from pyacemaker.factory import ModuleFactory
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
        self.state_file = Path(config.workflow.state_file_path)
        self.data_dir = Path(config.workflow.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.active_learning_dir = Path(config.workflow.active_learning_dir)
        self.active_learning_dir.mkdir(exist_ok=True)
        self.potentials_dir = Path(config.workflow.potentials_dir)
        self.potentials_dir.mkdir(exist_ok=True)

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
            # Create modules using factory
            self.generator, self.oracle, self.trainer, self.engine = ModuleFactory.create_modules(
                self.config
            )

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

    def _ensure_directory(self, path: Path) -> None:
        """
        Creates a directory and verifies write permissions.

        Args:
            path: Path to the directory.

        Raises:
            RuntimeError: If path exists but is not a directory.
            PermissionError: If directory is not writable.
            OSError: If directory creation fails.
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

    def _setup_iteration_directory(self, iteration: int) -> dict[str, Path]:
        """
        Creates the directory structure for the current iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Dictionary of paths for the iteration.
        """
        iter_dirname = TEMPLATE_ITER_DIR.format(iteration=iteration)
        iter_dir = self.active_learning_dir / iter_dirname
        paths = {
            "root": iter_dir,
            "candidates": iter_dir / "candidates",
            "dft_calc": iter_dir / "dft_calc",
            "training": iter_dir / "training",
            "md_run": iter_dir / "md_run",
        }

        for p in paths.values():
            self._ensure_directory(p)

        return paths

    def _explore(self, paths: dict[str, Path]) -> None:
        """
        Step 1: Exploration
        Generates candidate structures and saves them to the candidates directory.
        Uses streaming write to avoid opening/closing files repeatedly.
        """
        if not self.generator:
            return

        n_candidates = self.config.workflow.n_candidates
        batch_size = self.config.workflow.batch_size
        candidates_file = paths["candidates"] / FILENAME_CANDIDATES

        candidate_stream = self.generator.generate(n_candidates=n_candidates)
        total = 0

        # Open file once and append batches
        with candidates_file.open("a") as f:
            for batch in batched(candidate_stream, n=batch_size):
                # batched returns a tuple, ase.io.write accepts a sequence of Atoms.
                # No need to convert tuple to list.
                write(f, batch, format="extxyz")
                total += len(batch)

        self.logger.info(LOG_GENERATED_CANDIDATES.format(count=total))

    def _label(self, paths: dict[str, Path]) -> None:
        """
        Step 2: Labeling (Oracle)
        Reads candidate structures, computes properties, and saves labelled data.
        Uses streaming read and write.
        """
        if not self.oracle:
            return

        candidates_file = paths["candidates"] / FILENAME_CANDIDATES
        if not candidates_file.exists():
            self.logger.warning("No candidates found to label.")
            return

        batch_size = self.config.workflow.batch_size
        training_file = paths["training"] / FILENAME_TRAINING

        # Read from candidates file efficiently using streaming iterator
        candidate_stream = iread(str(candidates_file), index=":", format="extxyz")

        labelled_stream = self.oracle.compute(candidate_stream, batch_size=batch_size)
        total = 0

        # Open file once and append batches
        with training_file.open("a") as f:
            for batch in batched(labelled_stream, n=batch_size):
                write(f, batch, format="extxyz")
                total += len(batch)

        self.logger.info(LOG_COMPUTED_PROPERTIES.format(count=total))

    def _train(self, paths: dict[str, Path]) -> Path | None:
        """
        Step 3: Training
        Trains the potential using the labelled data.
        Returns path to the trained potential.
        """
        if not self.trainer:
            return None

        training_file = paths["training"] / FILENAME_TRAINING
        if not training_file.exists():
            self.logger.warning("No training data found, skipping training.")
            return None

        result = self.trainer.train(training_data_path=training_file)
        self.logger.info(LOG_POTENTIAL_TRAINED)

        # Assume result is a path to the potential file
        return Path(result) if isinstance(result, (str, Path)) else None

    def _deploy(self, paths: dict[str, Path], potential_path: Path | None) -> None:
        """
        Step 4: Deployment & Run
        Deploys the potential and runs MD/Engine.
        """
        if potential_path and potential_path.exists():
            filename = TEMPLATE_POTENTIAL_FILE.format(iteration=self.iteration)
            target_path = self.potentials_dir / filename
            with contextlib.suppress(shutil.SameFileError):
                shutil.copy(potential_path, target_path)
            self.logger.info(f"Deployed potential to {target_path}")

        if self.engine:
            # We assume Engine handles None structure/potential or finds them from config
            self.engine.run(structure=None, potential=None)
            self.logger.info(LOG_MD_COMPLETED)

    def _run_active_learning_step(self) -> None:
        """
        Executes a single step of the active learning loop using modular methods.
        """
        paths = self._setup_iteration_directory(self.iteration)

        self._explore(paths)
        self._label(paths)
        potential_path = self._train(paths)
        self._deploy(paths, potential_path)

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
