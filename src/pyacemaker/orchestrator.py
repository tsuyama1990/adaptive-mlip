import os
import shutil
from pathlib import Path

from ase import Atoms
from ase.io import iread, read, write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.loop import LoopState
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.defaults import (
    FILENAME_CANDIDATES,
    FILENAME_TRAINING,
    LOG_COMPUTED_PROPERTIES,
    LOG_GENERATED_CANDIDATES,
    LOG_INIT_MODULES,
    LOG_ITERATION_COMPLETED,
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

        self.state_file = Path(config.workflow.state_file_path)
        self.data_dir = Path(config.workflow.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.active_learning_dir = Path(config.workflow.active_learning_dir)
        self.active_learning_dir.mkdir(exist_ok=True)
        self.potentials_dir = Path(config.workflow.potentials_dir)
        self.potentials_dir.mkdir(exist_ok=True)

        # Core modules (placeholders)
        self.generator: BaseGenerator | None = None
        self.oracle: BaseOracle | None = None
        self.trainer: BaseTrainer | None = None
        self.engine: BaseEngine | None = None
        self.active_set_selector: ActiveSetSelector | None = None

        # Initialize State
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
            (
                self.generator,
                self.oracle,
                self.trainer,
                self.engine,
                self.active_set_selector,
            ) = ModuleFactory.create_modules(self.config)

        except Exception as e:
            self.logger.exception("Failed to initialize modules")
            raise RuntimeError(LOG_MODULE_INIT_FAIL.format(error=e)) from e

        self.logger.info(LOG_MODULES_INIT_SUCCESS)

    def load_state(self) -> None:
        """Loads the iteration state using LoopState."""
        try:
            self.loop_state = LoopState.load(self.state_file)
            self.logger.info(LOG_STATE_LOAD_SUCCESS.format(iteration=self.loop_state.iteration))
        except Exception as e:
            self.logger.warning(LOG_STATE_LOAD_FAIL.format(error=e))
            self.loop_state = LoopState()

    def save_state(self) -> None:
        """Saves the current iteration state."""
        try:
            self.loop_state.save(self.state_file)
            self.logger.debug(LOG_STATE_SAVED.format(state=self.loop_state.model_dump()))
        except Exception as e:
            self.logger.warning(LOG_STATE_SAVE_FAIL.format(error=e))

    def _ensure_directory(self, path: Path) -> None:
        """
        Creates a directory and verifies write permissions.
        Ensures partial directory creation is handled.
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
            # If creation failed, we should probably not attempt to remove anything automatically
            # as it might be a shared parent. But we log critical error.
            self.logger.critical(f"Failed to create directory {path}: {e}")
            raise

    def _setup_iteration_directory(self, iteration: int) -> dict[str, Path]:
        """
        Creates the directory structure for the current iteration.
        Attempts to clean up if partial creation fails.
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

        created_paths: list[Path] = []
        try:
            for p in paths.values():
                self._ensure_directory(p)
                created_paths.append(p)
        except Exception:
            # If any creation fails, log and re-raise.
            # We assume critical failure and let the workflow crash.
            # Cleanup might be dangerous if we delete pre-existing data.
            # So we just raise.
            self.logger.exception(f"Failed to setup iteration directory for iteration {iteration}")
            raise

        return paths

    def _explore(self, paths: dict[str, Path]) -> None:
        """Step 1: Exploration (Cold Start)"""
        if not self.generator:
            return

        n_candidates = self.config.workflow.n_candidates
        batch_size = self.config.workflow.batch_size
        candidates_file = paths["candidates"] / FILENAME_CANDIDATES

        candidate_stream = self.generator.generate(n_candidates=n_candidates)
        total = 0

        with candidates_file.open("a") as f:
            for batch in batched(candidate_stream, n=batch_size):
                write(f, batch, format="extxyz")
                total += len(batch)

        self.logger.info(LOG_GENERATED_CANDIDATES.format(count=total))

    def _label(self, paths: dict[str, Path]) -> None:
        """Step 2: Labeling (Oracle)"""
        if not self.oracle:
            return

        candidates_file = paths["candidates"] / FILENAME_CANDIDATES
        if not candidates_file.exists():
            self.logger.warning("No candidates found to label.")
            return

        batch_size = self.config.workflow.batch_size
        training_file = paths["training"] / FILENAME_TRAINING

        candidate_stream = iread(str(candidates_file), index=":", format="extxyz")

        labelled_stream = self.oracle.compute(candidate_stream, batch_size=batch_size)
        total = 0

        with training_file.open("a") as f:
            for batch in batched(labelled_stream, n=batch_size):
                write(f, batch, format="extxyz")
                total += len(batch)

        self.logger.info(LOG_COMPUTED_PROPERTIES.format(count=total))

    def _train(self, paths: dict[str, Path], initial_potential: Path | None = None) -> Path | None:
        """Step 3: Training"""
        if not self.trainer:
            return None

        training_file = paths["training"] / FILENAME_TRAINING
        if not training_file.exists():
            self.logger.warning("No training data found, skipping training.")
            return None

        result = self.trainer.train(training_data_path=training_file, initial_potential=initial_potential)
        self.logger.info(LOG_POTENTIAL_TRAINED)

        return Path(result) if isinstance(result, (str, Path)) else None

    def _check_initial_potential(self) -> None:
        """Checks if initial potential exists, if not generates one (Cold Start)."""
        if self.loop_state.current_potential and self.loop_state.current_potential.exists():
            self.logger.info(f"Using existing potential: {self.loop_state.current_potential}")
            return

        self.logger.info("No initial potential found. Starting Cold Start procedure.")

        # Use iteration 0 for cold start
        paths = self._setup_iteration_directory(0)

        self._explore(paths)
        self._label(paths)
        potential_path = self._train(paths)

        if potential_path:
            self.loop_state.current_potential = potential_path
            self.loop_state.iteration = 0
            self.save_state()
            self.logger.info(f"Cold start completed. Initial potential: {potential_path}")
        else:
            msg = "Cold start failed to produce a potential."
            raise RuntimeError(msg)

    def _get_initial_structure(self, iteration: int) -> Atoms | None:
        """Returns an initial structure for MD."""
        if not self.generator:
            return None

        try:
             # Try to get from candidates of previous iteration or iteration 0
             iter0_paths = self._setup_iteration_directory(0)
             cand_file = iter0_paths["candidates"] / FILENAME_CANDIDATES
             if cand_file.exists():
                 return next(iread(str(cand_file), index=0))

             # Fallback to generator
             return next(self.generator.generate(n_candidates=1))
        except Exception:
             self.logger.warning("Failed to get initial structure.")
             return None

    def _refine_potential(self, result: MDSimulationResult, potential_path: Path, paths: dict[str, Path]) -> Path | None:
        """Refines potential upon Halt."""
        if not result.halt_structure_path or not self.generator or not self.active_set_selector or not self.oracle or not self.trainer:
            return None

        # Check uncertainty threshold from config (Maintainability fix)
        # Note: Engine should ideally check this, but we can verify here or log
        threshold = self.config.workflow.otf.uncertainty_threshold
        if result.max_gamma <= threshold and not result.halted:
             # Should not be here if not halted, but safety check
             return None

        try:
            halt_structure = read(result.halt_structure_path)
            if isinstance(halt_structure, list):
                halt_structure = halt_structure[-1]

            # Generate local candidates
            local_n = self.config.workflow.otf.local_n_candidates
            candidates_gen = self.generator.generate_local(halt_structure, n_candidates=local_n)

            # Select Active Set
            n_select = self.config.workflow.otf.local_n_select
            selected_gen = self.active_set_selector.select(candidates_gen, potential_path, n_select=n_select)

            # Label
            labelled_gen = self.oracle.compute(selected_gen)

            # Append to training data
            training_file = paths["training"] / FILENAME_TRAINING
            count = 0
            batch_size = self.config.workflow.batch_size

            with training_file.open("a") as f:
                for batch in batched(labelled_gen, n=batch_size):
                    write(f, list(batch), format="extxyz")
                    count += len(batch)

            self.logger.info(f"Refinement: Added {count} new structures.")

            # Fine-tune
            return self._train(paths, initial_potential=potential_path)

        except Exception:
            self.logger.exception("Refinement failed")
            return None

    def _run_loop_iteration(self) -> None:
        """Executes one iteration of the active learning loop."""
        iteration = self.loop_state.iteration + 1
        paths = self._setup_iteration_directory(iteration)
        self.logger.info(LOG_START_ITERATION.format(iteration=iteration, max_iterations=self.config.workflow.max_iterations))

        # 1. Deploy Potential
        potential_filename = TEMPLATE_POTENTIAL_FILE.format(iteration=iteration)
        deployed_potential = self.potentials_dir / potential_filename

        if self.loop_state.current_potential:
             if self.loop_state.current_potential != deployed_potential:
                shutil.copy(self.loop_state.current_potential, deployed_potential)
        else:
            msg = "No current potential to deploy."
            raise RuntimeError(msg)

        # 2. Run MD
        initial_structure = self._get_initial_structure(iteration)
        if not initial_structure:
             self.logger.warning("No structure for MD. Skipping iteration.")
             # Update state safely
             self.loop_state.iteration = iteration
             self.save_state()
             return

        if self.engine:
            result = self.engine.run(structure=initial_structure, potential=deployed_potential)

            if result.halted:
                self.logger.info(f"MD Halted at step {result.n_steps}. Triggering refinement.")
                new_potential = self._refine_potential(result, deployed_potential, paths)
                if new_potential:
                    # Validate new potential path before updating state
                    if not new_potential.exists():
                        self.logger.error(f"Refined potential path {new_potential} does not exist!")
                    else:
                        self.loop_state.current_potential = new_potential
                        self.logger.info(f"Potential refined to: {new_potential}")
            else:
                 self.logger.info(LOG_ITERATION_COMPLETED.format(iteration=iteration))

        self.loop_state.iteration = iteration
        self.save_state()

    def _finalize(self) -> None:
        """Finalizes the workflow by deploying the best potential."""
        production_dir = Path("production")
        production_dir.mkdir(exist_ok=True)
        if self.loop_state.current_potential:
             target = production_dir / "potential.yace"
             shutil.copy(self.loop_state.current_potential, target)
             self.logger.info(f"Deployed best potential to {target}")

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        self.logger.info(LOG_START_LOOP)

        try:
            self.initialize_modules()
            self._check_initial_potential()

            while self.loop_state.iteration < self.config.workflow.max_iterations:
                 self._run_loop_iteration()

            self._finalize()
            self.logger.info(LOG_WORKFLOW_COMPLETED)

        except Exception as e:
            self.logger.critical(LOG_WORKFLOW_CRASHED.format(error=e))
            raise
