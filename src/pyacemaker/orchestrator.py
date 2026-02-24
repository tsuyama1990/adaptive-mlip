import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread, read, write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.directory_manager import DirectoryManager
from pyacemaker.core.exceptions import OrchestratorError
from pyacemaker.core.state_manager import StateManager
from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.defaults import (
    DEFAULT_PRODUCTION_DIR,
    FILENAME_CANDIDATES,
    FILENAME_POTENTIAL,
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
    LOG_WORKFLOW_COMPLETED,
    LOG_WORKFLOW_CRASHED,
    TEMPLATE_POTENTIAL_FILE,
)
from pyacemaker.domain_models.md import MDSimulationResult
from pyacemaker.factory import ModuleFactory
from pyacemaker.logger import setup_logger
from pyacemaker.utils.extraction import extract_local_region
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

        # Initialize Managers
        self.state_manager = StateManager(Path(config.workflow.state_file_path), self.logger)
        self.dir_manager = DirectoryManager(Path(config.workflow.active_learning_dir), self.logger)

        self.data_dir = Path(config.workflow.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.potentials_dir = Path(config.workflow.potentials_dir)
        self.potentials_dir.mkdir(exist_ok=True)

        # Core modules (placeholders)
        self.generator: BaseGenerator | None = None
        self.oracle: BaseOracle | None = None
        self.trainer: BaseTrainer | None = None
        self.engine: BaseEngine | None = None
        self.active_set_selector: ActiveSetSelector | None = None
        self.validator: Validator | None = None

        # Initialize State
        self.state_manager.load()
        self.logger.info(LOG_PROJECT_INIT.format(project_name=config.project_name))

    @property
    def loop_state(self) -> Any:
        # Expose loop_state for tests
        return self.state_manager.state

    def initialize_modules(self) -> None:
        """
        Initializes the core modules (Generator, Oracle, Trainer, Engine).

        Raises:
            OrchestratorError: If module initialization fails.
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
                self.validator,
            ) = ModuleFactory.create_modules(self.config)

        except Exception as e:
            self.logger.exception("Failed to initialize modules")
            msg = LOG_MODULE_INIT_FAIL.format(error=e)
            raise OrchestratorError(msg) from e

        self.logger.info(LOG_MODULES_INIT_SUCCESS)

    def _stream_write(
        self,
        generator: Iterable[Atoms],
        filepath: Path,
        batch_size: int = 100,
        append: bool = False,
    ) -> int:
        """
        Writes atoms from a generator to a file in chunks to balance I/O and memory.
        Uses `batched` to process chunks.
        Optimized to open file once.

        Args:
            generator: Iterable of Atoms objects.
            filepath: Path to output file.
            batch_size: Number of atoms per write operation.
            append: Whether to append to the file or overwrite.

        Returns:
            Total number of atoms written.
        """
        count = 0

        # Ensure parent dir exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"

        # Open file once for the entire batch processing to avoid repeated I/O overhead
        with filepath.open(mode) as f:
            for batch in batched(generator, batch_size):
                # Write chunk
                # ase.io.write handles list of atoms and can write to file handle.
                # format="extxyz" is standard for streaming.
                write(f, batch, format="extxyz")
                count += len(batch)

        return count

    def _explore(self, paths: dict[str, Path]) -> None:
        """
        Step 1: Exploration (Cold Start).
        Generates initial candidate structures and writes them to disk using efficient streaming.
        """
        if not self.generator:
            return

        n_candidates = self.config.workflow.n_candidates
        candidates_file = paths["candidates"] / FILENAME_CANDIDATES

        try:
            candidate_stream = self.generator.generate(n_candidates=n_candidates)
            # Use explicit chunked streaming
            total = self._stream_write(
                candidate_stream,
                candidates_file,
                batch_size=self.config.workflow.batch_size,
                append=True
            )

            self.logger.info(LOG_GENERATED_CANDIDATES.format(count=total))
        except Exception as e:
            msg = f"Exploration failed: {e}"
            raise OrchestratorError(msg) from e

    def _label(self, paths: dict[str, Path]) -> None:
        """
        Step 2: Labeling (Oracle).
        Computes properties for candidates and writes labelled data to training set.
        """
        if not self.oracle:
            return

        candidates_file = paths["candidates"] / FILENAME_CANDIDATES
        if not candidates_file.exists():
            self.logger.warning("No candidates found to label.")
            return

        batch_size = self.config.workflow.batch_size
        training_file = paths["training"] / FILENAME_TRAINING

        try:
            # Lazy read of candidates
            candidate_stream = iread(str(candidates_file), index=":", format="extxyz")

            # Streaming computation
            labelled_stream = self.oracle.compute(candidate_stream, batch_size=batch_size)

            total = self._stream_write(
                labelled_stream,
                training_file,
                batch_size=batch_size,
                append=True
            )

            self.logger.info(LOG_COMPUTED_PROPERTIES.format(count=total))
        except Exception as e:
            msg = f"Labeling failed: {e}"
            raise OrchestratorError(msg) from e

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
        if self.state_manager.current_potential and self.state_manager.current_potential.exists():
            self.logger.info(f"Using existing potential: {self.state_manager.current_potential}")
            return

        self.logger.info("No initial potential found. Starting Cold Start procedure.")

        # Use iteration 0 for cold start
        paths = self.dir_manager.setup_iteration(0)

        self._explore(paths)
        self._label(paths)
        potential_path = self._train(paths)

        if potential_path:
            self.state_manager.current_potential = potential_path
            self.state_manager.iteration = 0
            self.state_manager.save()
            self.logger.info(f"Cold start completed. Initial potential: {potential_path}")
        else:
            msg = "Cold start failed to produce a potential."
            raise OrchestratorError(msg)

    def _get_initial_structure(self, iteration: int) -> Atoms | None:
        """Returns an initial structure for MD."""
        if not self.generator:
            return None

        try:
             # Try to get from candidates of previous iteration or iteration 0
             iter0_paths = self.dir_manager.setup_iteration(0)
             cand_file = iter0_paths["candidates"] / FILENAME_CANDIDATES
             if cand_file.exists():
                 # Use next() on iread to get just the first frame efficiently
                 return next(iread(str(cand_file), index=0))

             # Fallback to generator
             return next(self.generator.generate(n_candidates=1))
        except Exception:
             self.logger.warning("Failed to get initial structure.")
             return None

    def _get_max_gamma_atom_index(self, structure: Atoms) -> int:
        """Finds the index of the atom with the maximum gamma value."""
        if "c_gamma" in structure.arrays:
            gammas = structure.get_array("c_gamma")  # type: ignore[no-untyped-call]
            return int(np.argmax(gammas))

        self.logger.warning("c_gamma not found in structure arrays. Using atom 0 as center.")
        return 0

    def _extract_cluster(self, halt_structure_path: str) -> Atoms | None:
        """
        Loads the halt structure and extracts the local cluster around the highest uncertainty atom.
        """
        try:
            halt_structure = read(halt_structure_path)
            if isinstance(halt_structure, list):
                halt_structure = halt_structure[-1]

            # Find center atom (max gamma)
            center_idx = self._get_max_gamma_atom_index(halt_structure)

            # Extract local cluster (S0)
            radius = self.config.structure.local_extraction_radius
            buffer = self.config.structure.local_buffer_radius

            return extract_local_region(halt_structure, center_idx, radius, buffer)
        except Exception:
            self.logger.exception("Failed to extract local cluster.")
            return None

    def _select_and_label(
        self,
        s0_cluster: Atoms,
        potential_path: Path,
        paths: dict[str, Path]
    ) -> int:
        """
        Generates local candidates, selects active set, labels them, and writes to training file.
        Returns the number of structures added.
        """
        if not self.generator or not self.active_set_selector or not self.oracle:
            return 0

        # Generate local candidates (perturbations of S0)
        local_n = self.config.workflow.otf.local_n_candidates

        # Pass engine and potential for advanced local generation strategies (e.g. MD Micro Burst)
        candidates_gen = self.generator.generate_local(
            s0_cluster,
            n_candidates=local_n,
            engine=self.engine,
            potential=potential_path
        )

        # Select Active Set (including S0 as anchor)
        n_select = self.config.workflow.otf.local_n_select
        selected_gen = self.active_set_selector.select(
            candidates_gen,
            potential_path,
            n_select=n_select,
            anchor=s0_cluster
        )

        # Label
        labelled_gen = self.oracle.compute(selected_gen)

        # Append to training data
        training_file = paths["training"] / FILENAME_TRAINING
        batch_size = self.config.workflow.batch_size

        return self._stream_write(
            labelled_gen,
            training_file,
            batch_size=batch_size,
            append=True
        )

    def _refine_potential(self, result: MDSimulationResult, potential_path: Path, paths: dict[str, Path]) -> Path | None:
        """
        Refines potential upon Halt.
        Orchestrates extraction, selection, labeling, and retraining.
        """
        if not result.halt_structure_path or not self.generator or not self.active_set_selector or not self.oracle or not self.trainer:
            return None

        threshold = self.config.workflow.otf.uncertainty_threshold
        if result.max_gamma <= threshold and not result.halted:
             return None

        try:
            s0_cluster = self._extract_cluster(result.halt_structure_path)
            if s0_cluster is None:
                return None

            count = self._select_and_label(s0_cluster, potential_path, paths)
            self.logger.info(f"Refinement: Added {count} new structures.")

            # Fine-tune
            return self._train(paths, initial_potential=potential_path)

        except Exception:
            self.logger.exception("Refinement failed")
            return None

    def _deploy_potential(self, iteration: int) -> Path:
        """Deploys the current potential to the potentials directory."""
        potential_filename = TEMPLATE_POTENTIAL_FILE.format(iteration=iteration)
        deployed_potential = self.potentials_dir / potential_filename

        if self.state_manager.current_potential:
             if self.state_manager.current_potential != deployed_potential:
                shutil.copy(self.state_manager.current_potential, deployed_potential)
        else:
            msg = "No current potential to deploy."
            raise OrchestratorError(msg)

        return deployed_potential

    def _run_md_simulation(self, iteration: int, deployed_potential: Path) -> MDSimulationResult | None:
        """Runs the MD simulation."""
        initial_structure = self._get_initial_structure(iteration)
        if not initial_structure:
             self.logger.warning("No structure for MD. Skipping iteration.")
             return None

        if self.engine:
            return self.engine.run(structure=initial_structure, potential=deployed_potential)
        return None

    def _handle_md_halt(self, result: MDSimulationResult, deployed_potential: Path, paths: dict[str, Path]) -> None:
        """Handles MD halt logic and triggers refinement."""
        if result.halted:
            self.logger.info(f"MD Halted at step {result.n_steps}. Triggering refinement.")
            new_potential = self._refine_potential(result, deployed_potential, paths)
            if new_potential:
                if not new_potential.exists():
                    self.logger.error(f"Refined potential path {new_potential} does not exist!")
                else:
                    self.state_manager.current_potential = new_potential
                    self.logger.info(f"Potential refined to: {new_potential}")
        else:
             self.logger.info(LOG_ITERATION_COMPLETED.format(iteration=self.state_manager.iteration + 1))

    def _adapt_strategy(self, result: MDSimulationResult) -> None:
        """
        Placeholder for adaptive policy logic.
        Future implementation: Update self.generator.config based on result metrics.
        """

    def _run_loop_iteration(self) -> None:
        """Executes one iteration of the active learning loop."""
        iteration = self.state_manager.iteration + 1
        paths = self.dir_manager.setup_iteration(iteration)
        self.logger.info(LOG_START_ITERATION.format(iteration=iteration, max_iterations=self.config.workflow.max_iterations))

        try:
            deployed_potential = self._deploy_potential(iteration)
            result = self._run_md_simulation(iteration, deployed_potential)

            if result:
                self._adapt_strategy(result)
                self._handle_md_halt(result, deployed_potential, paths)

            self.state_manager.iteration = iteration
            self.state_manager.save()

        except Exception as e:
            self.logger.exception(f"Iteration {iteration} failed")
            msg = f"Iteration {iteration} failed: {e}"
            raise OrchestratorError(msg) from e

    def _finalize(self) -> None:
        """Finalizes the workflow by deploying and validating the best potential."""
        production_dir = Path(DEFAULT_PRODUCTION_DIR)
        production_dir.mkdir(exist_ok=True)
        potential_target = production_dir / FILENAME_POTENTIAL

        if self.state_manager.current_potential:
            shutil.copy(self.state_manager.current_potential, potential_target)
            self.logger.info(f"Deployed best potential to {potential_target}")

            if self.validator:
                report_path = production_dir / "validation_report.html"
                # Get a structure for validation. Use initial structure of last iteration.
                structure = self._get_initial_structure(self.state_manager.iteration)

                if structure:
                    self.logger.info("Running final validation...")
                    result = self.validator.validate(potential_target, report_path, structure)
                    status = (
                        "PASSED"
                        if (result.phonon_stable and result.elastic_stable)
                        else "FAILED"
                    )
                    self.logger.info(f"Validation {status}. Report saved to {report_path}")
                else:
                    self.logger.warning("Could not retrieve structure for validation.")

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        self.logger.info(LOG_START_LOOP)

        try:
            self.initialize_modules()
            self._check_initial_potential()

            while self.state_manager.iteration < self.config.workflow.max_iterations:
                 self._run_loop_iteration()

            self._finalize()
            self.logger.info(LOG_WORKFLOW_COMPLETED)

        except Exception as e:
            self.logger.critical(LOG_WORKFLOW_CRASHED.format(error=e))
            raise
