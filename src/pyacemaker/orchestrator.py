import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread, read, write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.directory_manager import DirectoryManager
from pyacemaker.core.exceptions import OrchestratorError
from pyacemaker.core.loop import LoopStatus
from pyacemaker.core.state_manager import StateManager
from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.data import AtomStructure
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
    LOG_STEP_1,
    LOG_STEP_2,
    LOG_STEP_3,
    LOG_STEP_4,
    LOG_STEP_5,
    LOG_STEP_6,
    LOG_STEP_7,
    LOG_WORKFLOW_COMPLETED,
    LOG_WORKFLOW_CRASHED,
    TEMPLATE_POTENTIAL_FILE,
    WORKFLOW_MODE_DISTILLATION,
)
from pyacemaker.domain_models.md import MDSimulationResult
from pyacemaker.domain_models.workflow import WorkflowStep
from pyacemaker.factory import ModuleFactory
from pyacemaker.logger import setup_logger
from pyacemaker.modules.sampling import DirectSampler
from pyacemaker.utils.extraction import extract_local_region


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
        self.state_manager = StateManager(
            Path(config.workflow.state_file_path),
            self.logger,
            checkpoint_interval=config.workflow.checkpoint_interval,
        )
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
        generator: Iterable[AtomStructure],
        filepath: Path,
        batch_size: int = 100,
        append: bool = False,
    ) -> int:
        """
        Writes AtomStructure from a generator to a file in chunks to balance I/O and memory.

        This method implements buffered writing to ensure scalability for large datasets (e.g., 1M structures).
        It consumes the generator lazily, accumulating only `batch_size` items in memory at a time.
        The file is kept open during the process to minimize I/O overhead.

        Args:
            generator: Iterable of AtomStructure objects.
            filepath: Path to output file.
            batch_size: Number of atoms to buffer in memory before writing chunk.
            append: Whether to append to the file or overwrite.

        Returns:
            Total number of structures written.
        """
        count = 0

        # Ensure parent dir exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"

        if batch_size <= 0:
            raise OrchestratorError("Batch size must be positive")

        # Buffer a chunk of Atoms objects to write at once using ASE's list write support
        # This reduces I/O calls compared to writing one by one.
        buffer: list[Atoms] = []

        try:
            with filepath.open(mode) as f:
                for structure in generator:
                    # Convert to ASE Atoms with metadata
                    atoms = structure.to_ase()
                    buffer.append(atoms)

                    if len(buffer) >= batch_size:
                        write(f, buffer, format="extxyz")
                        count += len(buffer)
                        buffer.clear()

                # Write remaining
                if buffer:
                    write(f, buffer, format="extxyz")
                    count += len(buffer)
                    buffer.clear()
        except OSError as e:
            msg = f"I/O error during streaming write to {filepath}: {e}"
            raise OrchestratorError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during streaming write to {filepath}: {e}"
            raise OrchestratorError(msg) from e

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
            # Get iterator from generator
            candidate_stream = self.generator.generate(n_candidates=n_candidates)

            # _stream_write consumes the iterator, buffering in chunks.
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
            # Note: iread returns Iterator[Atoms]
            # We need to wrap them into Iterator[AtomStructure] for the Oracle

            def atom_structure_generator(file_path: str) -> Iterator[AtomStructure]:
                for atoms in iread(file_path, index=":", format="extxyz"):
                    if isinstance(atoms, Atoms):
                        yield AtomStructure.from_ase(atoms)

            candidate_stream = atom_structure_generator(str(candidates_file))

            # Streaming computation
            # Oracle now returns Iterator[AtomStructure]
            labelled_stream = self.oracle.compute(candidate_stream, batch_size=batch_size)

            # _stream_write now accepts Iterator[AtomStructure]
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

        try:
            result = self.trainer.train(training_data_path=training_file, initial_potential=initial_potential)
            self.logger.info(LOG_POTENTIAL_TRAINED)
            return Path(result) if isinstance(result, (str, Path)) else None
        except Exception as e:
            msg = f"Training failed: {e}"
            raise OrchestratorError(msg) from e

    def _check_initial_potential(self) -> None:
        """Checks if initial potential exists, if not generates one (Cold Start)."""
        if self.state_manager.state.current_potential and self.state_manager.state.current_potential.exists():
            self.logger.info(f"Using existing potential: {self.state_manager.state.current_potential}")
            return

        self.logger.info("No initial potential found. Starting Cold Start procedure.")

        # Use iteration 0 for cold start
        paths = self.dir_manager.setup_iteration(0)

        self._explore(paths)
        self._label(paths)
        potential_path = self._train(paths)

        if potential_path:
            self.state_manager.state.current_potential = potential_path
            self.state_manager.state.iteration = 0
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
             # Generator.generate returns Iterator[AtomStructure]
             # We need to return Atoms
             gen = self.generator.generate(n_candidates=1)
             try:
                 struct = next(gen)
                 return struct.to_ase()
             except StopIteration:
                 return None

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

    def _generate_local_candidates(
        self, s0_cluster: Atoms, potential_path: Path
    ) -> Iterator[AtomStructure]:
        """Generates local candidates using the configured strategy."""
        if not self.generator:
            return iter([])

        local_n = self.config.workflow.otf.local_n_candidates
        return self.generator.generate_local(
            s0_cluster,
            n_candidates=local_n,
            engine=self.engine,
            potential=potential_path,
        )

    def _execute_active_learning_selection(
        self,
        candidates_gen: Iterator[AtomStructure],
        s0_cluster: Atoms,
        potential_path: Path,
    ) -> Iterator[AtomStructure]:
        """Selects the active set from candidates."""
        if not self.active_set_selector:
            return iter([])

        def to_ase_iter(struct_iter: Iterator[AtomStructure]) -> Iterator[Atoms]:
            for s in struct_iter:
                yield s.to_ase()

        def atom_structure_adapter(atoms_iter: Iterable[Atoms]) -> Iterator[AtomStructure]:
            for atoms in atoms_iter:
                yield AtomStructure.from_ase(atoms)

        n_select = self.config.workflow.otf.local_n_select
        candidates_ase_gen = to_ase_iter(candidates_gen)

        selected_ase_gen = self.active_set_selector.select(
            candidates_ase_gen,  # type: ignore[arg-type]
            potential_path,
            n_select=n_select,
            anchor=s0_cluster,
        )
        return atom_structure_adapter(selected_ase_gen)

    def _select_and_label(
        self, s0_cluster: Atoms, potential_path: Path, paths: dict[str, Path]
    ) -> int:
        """
        Generates local candidates, selects active set, labels them, and writes to training file.
        Returns the number of structures added.
        """
        if not self.generator or not self.active_set_selector or not self.oracle:
            return 0

        candidates_gen = self._generate_local_candidates(s0_cluster, potential_path)
        selected_struct_gen = self._execute_active_learning_selection(
            candidates_gen, s0_cluster, potential_path
        )
        labelled_gen = self.oracle.compute(selected_struct_gen)

        training_file = paths["training"] / FILENAME_TRAINING
        batch_size = self.config.workflow.batch_size

        return self._stream_write(
            labelled_gen, training_file, batch_size=batch_size, append=True
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

        except Exception as e:
            msg = f"Refinement failed: {e}"
            self.logger.exception(msg)
            raise OrchestratorError(msg) from e

    def _deploy_potential(self, iteration: int) -> Path:
        """Deploys the current potential to the potentials directory."""
        potential_filename = TEMPLATE_POTENTIAL_FILE.format(iteration=iteration)
        deployed_potential = self.potentials_dir / potential_filename

        if self.state_manager.state.current_potential:
             if self.state_manager.state.current_potential != deployed_potential:
                shutil.copy(self.state_manager.state.current_potential, deployed_potential)
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
            try:
                return self.engine.run(structure=initial_structure, potential=deployed_potential)
            except Exception as e:
                raise OrchestratorError(f"MD Simulation failed: {e}") from e
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
                    self.state_manager.state.current_potential = new_potential
                    self.logger.info(f"Potential refined to: {new_potential}")
        else:
             self.logger.info(LOG_ITERATION_COMPLETED.format(iteration=self.state_manager.state.iteration + 1))

    def _adapt_strategy(self, result: MDSimulationResult) -> None:
        """
        Placeholder for adaptive policy logic.
        Future implementation: Update self.generator.config based on result metrics.

        Note: This method is intended to implement the "Adaptive Exploration Policy" described in the Spec.
        Currently, it is a no-op as the complex adaptation logic requires further requirements analysis.
        """

    def _execute_iteration_logic(self, iteration: int, paths: dict[str, Path]) -> None:
        """
        Core logic for a single iteration.
        Separated for clarity and testability.
        """
        deployed_potential = self._deploy_potential(iteration)
        result = self._run_md_simulation(iteration, deployed_potential)

        if result:
            self._adapt_strategy(result)
            self._handle_md_halt(result, deployed_potential, paths)

    def _run_loop_iteration(self) -> None:
        """Executes one iteration of the active learning loop."""
        iteration = self.state_manager.state.iteration + 1
        paths = self.dir_manager.setup_iteration(iteration)
        self.logger.info(LOG_START_ITERATION.format(iteration=iteration, max_iterations=self.config.workflow.max_iterations))

        try:
            self._execute_iteration_logic(iteration, paths)

            self.state_manager.state.iteration = iteration
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

        if self.state_manager.state.current_potential:
            shutil.copy(self.state_manager.state.current_potential, potential_target)
            self.logger.info(f"Deployed best potential to {potential_target}")

            if self.validator:
                report_path = production_dir / "validation_report.html"
                # Get a structure for validation. Use initial structure of last iteration.
                structure = self._get_initial_structure(self.state_manager.state.iteration)

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

    def _run_legacy_loop(self) -> None:
        """Executes the legacy active learning loop."""
        self._check_initial_potential()

        while self.state_manager.state.iteration < self.config.workflow.max_iterations:
            self._run_loop_iteration()

    def _step1_direct_sampling(self) -> None:
        self.logger.info(LOG_STEP_1)
        if not self.generator:
            raise OrchestratorError("Generator not initialized")

        # Refactor: Ensure step 1 config is accessed safely
        if not self.config.distillation.step1_direct_sampling:
             raise OrchestratorError("Step 1 configuration missing")

        config = self.config.distillation.step1_direct_sampling

        step_dir = self.dir_manager.base_dir / "step1_direct_sampling"
        step_dir.mkdir(parents=True, exist_ok=True)
        candidates_file = step_dir / FILENAME_CANDIDATES

        try:
            # Instantiate DirectSampler with the underlying generator
            sampler = DirectSampler(config, self.generator)

            # Use sampler to generate selected structures
            stream = sampler.generate()

            count = self._stream_write(
                stream, candidates_file, batch_size=config.batch_size
            )
            self.logger.info(LOG_GENERATED_CANDIDATES.format(count=count))
        except Exception as e:
            raise OrchestratorError(f"Step 1 failed: {e}") from e

    def _step2_active_learning(self) -> None:
        self.logger.info(LOG_STEP_2)
        if not self.oracle:
            raise OrchestratorError("Oracle not initialized")

        config = self.config.distillation.step2_active_learning
        if config is None:
            raise OrchestratorError("Step 2 configuration missing")

        # Paths
        step1_dir = self.dir_manager.base_dir / "step1_direct_sampling"
        candidates_file = step1_dir / FILENAME_CANDIDATES

        step2_dir = self.dir_manager.base_dir / "step2_active_learning"
        step2_dir.mkdir(parents=True, exist_ok=True)

        if not candidates_file.exists():
             raise OrchestratorError(f"Step 1 output not found: {candidates_file}")

        try:
            # 1. Read candidates from Step 1
            def atom_structure_generator(file_path: str) -> Iterator[AtomStructure]:
                # Using iread to potentially stream, though here we need sorting
                for atoms in iread(file_path, index=":", format="extxyz"):
                    if isinstance(atoms, Atoms):
                        yield AtomStructure.from_ase(atoms)

            # 2. Compute Uncertainty using MaceOracle
            candidates_iter = atom_structure_generator(str(candidates_file))

            # Using configurable batch size
            scored_candidates = list(self.oracle.compute(candidates_iter, batch_size=config.batch_size))

            if not scored_candidates:
                self.logger.warning("No candidates scored by Oracle.")
                return

            # 3. Filter based on uncertainty (Memory Efficient Top-K)
            # Instead of sorting the entire list (which is O(N log N) and memory heavy),
            # we use heapq.nlargest to find the top k elements in O(N log k).
            import heapq

            # Wrapper class for heapq to handle None uncertainties and reverse ordering
            class ScoredItem:
                def __init__(self, struct: AtomStructure) -> None:
                    self.struct = struct
                    # Default to -inf if None to place at bottom
                    self.score = struct.uncertainty if struct.uncertainty is not None else -float('inf')

                def __lt__(self, other: "ScoredItem") -> bool:
                    return self.score < other.score

            top_k_items = heapq.nlargest(
                config.n_active,
                [ScoredItem(c) for c in scored_candidates]
            )

            active_set: list[AtomStructure] = []
            for item in top_k_items:
                cand = item.struct
                if cand.uncertainty is not None and cand.uncertainty >= config.uncertainty_threshold:
                     active_set.append(cand)
                else:
                     # Prioritize threshold but fill up to n_active if needed
                     active_set.append(cand)

            if not active_set:
                self.logger.warning("No candidates met active set selection criteria.")
                return

            # 4. Save Active Set
            active_set_file = step2_dir / "active_set.xyz"

            self._stream_write(
                iter(active_set),
                active_set_file,
                batch_size=len(active_set)
            )

            self.logger.info(f"Selected {len(active_set)} structures for active set. Saved to {active_set_file}")

        except Exception as e:
            raise OrchestratorError(f"Step 2 failed: {e}") from e

    def _step3_mace_finetune(self) -> None:
        self.logger.info(LOG_STEP_3)
        if not self.trainer:
            raise OrchestratorError("Trainer not initialized")
        # Placeholder for MACE Fine-tuning

    def _step4_surrogate_sampling(self) -> None:
        self.logger.info(LOG_STEP_4)
        if not self.generator:
            raise OrchestratorError("Generator not initialized")
        # Placeholder for Surrogate Data Generation

    def _step5_surrogate_labeling(self) -> None:
        self.logger.info(LOG_STEP_5)
        if not self.oracle:
            raise OrchestratorError("Oracle not initialized")
        # Placeholder for Surrogate Labeling

    def _step6_pacemaker_base(self) -> None:
        self.logger.info(LOG_STEP_6)
        if not self.trainer:
            raise OrchestratorError("Trainer not initialized")
        # Placeholder for Pacemaker Base Training

    def _step7_delta_learning(self) -> None:
        self.logger.info(LOG_STEP_7)
        if not self.trainer:
            raise OrchestratorError("Trainer not initialized")

        # This is the critical step.
        # We would use self.trainer to train on DFT data.
        # For now, we simulate success.
        step_dir = self.dir_manager.base_dir / "step7_delta_learning"
        step_dir.mkdir(parents=True, exist_ok=True)

    def _run_distillation_workflow(self) -> None:
        """Executes the 7-step MACE Distillation workflow."""
        steps = [
            (WorkflowStep.DIRECT_SAMPLING, self._step1_direct_sampling),
            (WorkflowStep.ACTIVE_LEARNING, self._step2_active_learning),
            (WorkflowStep.MACE_FINETUNE, self._step3_mace_finetune),
            (WorkflowStep.SURROGATE_SAMPLING, self._step4_surrogate_sampling),
            (WorkflowStep.SURROGATE_LABELING, self._step5_surrogate_labeling),
            (WorkflowStep.PACEMAKER_BASE, self._step6_pacemaker_base),
            (WorkflowStep.DELTA_LEARNING, self._step7_delta_learning),
        ]

        self.state_manager.state.mode = WORKFLOW_MODE_DISTILLATION

        # Determine start index based on current step in state
        start_index = 0
        if self.state_manager.state.current_step:
            for i, (step_enum, _) in enumerate(steps):
                if step_enum == self.state_manager.state.current_step:
                    # Resume from the found step
                    start_index = i
                    break

        for i in range(start_index, len(steps)):
            step_enum, step_func = steps[i]
            self.state_manager.state.current_step = step_enum
            self.state_manager.save(force=True)

            try:
                step_func()
            except Exception as e:
                self.logger.error(f"Failed at step {step_enum}: {e}")
                # Attempt rollback to previous step state before halting
                self.state_manager.rollback()
                self.state_manager.state.status = LoopStatus.HALTED
                self.state_manager.save(force=True)
                raise OrchestratorError(f"Workflow halted due to error in {step_enum}: {e}") from e

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        self.logger.info(LOG_START_LOOP)

        try:
            self.initialize_modules()

            if self.config.distillation.enable_mace_distillation:
                self._run_distillation_workflow()
            else:
                self._run_legacy_loop()

            self._finalize()
            self.logger.info(LOG_WORKFLOW_COMPLETED)

        except Exception as e:
            self.logger.critical(LOG_WORKFLOW_CRASHED.format(error=e))
            raise OrchestratorError(f"Run failed: {e}") from e
