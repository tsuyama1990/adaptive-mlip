import contextlib
import json
import os
import shutil
from pathlib import Path

from ase import Atoms
from ase.io import iread, read, write

from pyacemaker.core.active_set import ActiveSetSelector
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
from pyacemaker.domain_models.md import MDSimulationResult
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
        self.active_set_selector: ActiveSetSelector | None = None

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
            self.active_set_selector = ActiveSetSelector()

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

    def _handle_halt_event(  # noqa: PLR0911
        self,
        result: MDSimulationResult,
        potential_path: Path,
        paths: dict[str, Path]
    ) -> Path | None:
        """
        Handles an MD halt event by generating local candidates, selecting active set,
        computing properties, and fine-tuning the potential.
        """
        if not (self.generator and self.oracle and self.trainer and self.active_set_selector):
             self.logger.error("Modules not fully initialized.")
             return None

        if not result.halt_structure_path:
             self.logger.error("No halt structure path in result.")
             return None

        try:
            halt_structure = read(result.halt_structure_path)
            if isinstance(halt_structure, list):
                halt_structure = halt_structure[-1]
            if not isinstance(halt_structure, Atoms):
                 self.logger.error("Halt structure is not an Atoms object.")
                 return None
        except Exception:
            self.logger.exception("Failed to read halt structure")
            return None

        # Generate Local Candidates
        local_n = self.config.workflow.otf.local_n_candidates
        candidates_gen = self.generator.generate_local(halt_structure, n_candidates=local_n)

        # Select Active Set
        n_select = self.config.workflow.otf.local_n_select
        try:
             # Generator returns iterator, so we can pass it directly.
             # Note: select returns iterator too.
             selected_gen = self.active_set_selector.select(
                 candidates_gen,
                 potential_path,
                 n_select=n_select
             )
        except Exception:
             self.logger.exception("Active set selection failed")
             return None

        # Compute properties (Oracle)
        labelled_gen = self.oracle.compute(selected_gen)

        # Append to training data
        training_file = paths["training"] / FILENAME_TRAINING
        count = 0
        with training_file.open("a") as f:
             for batch in batched(labelled_gen, n=1):
                 write(f, batch, format="extxyz")
                 count += len(batch)

        if count == 0:
             self.logger.warning("No new training data generated from halt event.")
             return None

        self.logger.info(f"Added {count} new structures from OTF loop.")

        # Fine-tune the potential
        new_potential = self.trainer.train(training_file, initial_potential=potential_path)

        return Path(new_potential) if isinstance(new_potential, (str, Path)) else None

    def _run_otf_loop(self, paths: dict[str, Path], potential_path: Path | None) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Step 4: Deployment & Run (OTF Loop)
        Deploys potential, runs MD, and handles active learning events.
        """
        if not potential_path or not potential_path.exists():
            self.logger.warning("No potential to deploy/run MD.")
            return

        # Deploy potential
        filename = TEMPLATE_POTENTIAL_FILE.format(iteration=self.iteration)
        target_path = self.potentials_dir / filename
        with contextlib.suppress(shutil.SameFileError):
            shutil.copy(potential_path, target_path)
        self.logger.info(f"Deployed potential to {target_path}")

        if not self.engine:
            return

        # Load initial structure for MD
        # Try to use first candidate from current iteration
        candidates_file = paths["candidates"] / FILENAME_CANDIDATES
        initial_structure = None
        try:
            if candidates_file.exists():
                initial_structure = next(iread(str(candidates_file), index=0))
        except (StopIteration, Exception):
            self.logger.warning("Could not load initial structure from candidates.")

        if initial_structure is None:
             self.logger.warning("No valid initial structure for MD. Skipping simulation.")
             return

        current_potential = potential_path
        retries = 0
        max_retries = self.config.workflow.otf.max_retries

        while retries <= max_retries:
            self.logger.info(f"Starting MD run (attempt {retries}/{max_retries})")

            try:
                result = self.engine.run(structure=initial_structure, potential=current_potential)
            except Exception:
                self.logger.exception("MD execution failed")
                break

            if not result.halted:
                self.logger.info(LOG_MD_COMPLETED)
                break

            self.logger.warning(
                f"MD halted at step {result.n_steps} (Gamma: {result.max_gamma:.2f} > {self.config.workflow.otf.uncertainty_threshold})"
            )

            if retries >= max_retries:
                self.logger.error("Max OTF retries reached. Moving to next iteration.")
                break

            # Handle Halt
            new_potential = self._handle_halt_event(result, current_potential, paths)

            if not new_potential:
                self.logger.warning("Refinement failed. Aborting OTF loop.")
                break

            current_potential = new_potential

            # Update initial structure for next run to continue from halt?
            # Usually we want to continue, but LammpsEngine.run starts fresh.
            # So we pass the halted structure as new initial structure.
            if result.halt_structure_path:
                try:
                    new_init = read(result.halt_structure_path)
                    if isinstance(new_init, list):
                        new_init = new_init[-1]
                    if isinstance(new_init, Atoms):
                        initial_structure = new_init
                    else:
                        self.logger.error("Resume structure is invalid.")
                        break
                except Exception:
                    self.logger.exception("Failed to read halt structure for resume")
                    break

            retries += 1

    def _run_active_learning_step(self) -> None:
        """
        Executes a single step of the active learning loop using modular methods.
        """
        paths = self._setup_iteration_directory(self.iteration)

        self._explore(paths)
        self._label(paths)
        potential_path = self._train(paths)
        self._run_otf_loop(paths, potential_path)

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
