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
    LOG_INIT_MODULES,
    LOG_ITERATION_COMPLETED,
    LOG_MODULE_INIT_FAIL,
    LOG_MODULES_INIT_SUCCESS,
    LOG_PROJECT_INIT,
    LOG_START_ITERATION,
    LOG_START_LOOP,
    LOG_WORKFLOW_COMPLETED,
    LOG_WORKFLOW_CRASHED,
    TEMPLATE_POTENTIAL_FILE,
)
from pyacemaker.factory import ModuleFactory
from pyacemaker.logger import setup_logger
from pyacemaker.utils.extraction import extract_intelligent_cluster


class Orchestrator:
    """
    Central controller for the PYACEMAKER workflow.
    Manages the lifecycle of the active learning loop following the 4-Phase
    Hierarchical Distillation Architecture.
    """

    def __init__(self, config: PyAceConfig) -> None:
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
        return self.state_manager.state

    def initialize_modules(self) -> None:
        self.logger.info(LOG_INIT_MODULES)
        try:
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
        count = 0
        filepath.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        with filepath.open(mode) as f:
            for atoms in generator:
                write(f, atoms, format="extxyz")
                count += 1
        return count

    def _get_initial_structure(self, iteration: int) -> Atoms | None:
        if not self.generator:
            return None
        try:
            iter0_paths = self.dir_manager.setup_iteration(0)
            cand_file = iter0_paths["candidates"] / FILENAME_CANDIDATES
            if cand_file.exists():
                return next(iread(str(cand_file), index=0))
            return next(self.generator.generate(n_candidates=1))
        except Exception:
            self.logger.warning("Failed to get initial structure.")
            return None

    def _get_max_gamma_atom_index(self, structure: Atoms) -> int:
        if "c_gamma" in structure.arrays:
            gammas = structure.get_array("c_gamma")  # type: ignore[no-untyped-call]
            return int(np.argmax(gammas))
        return 0

    def _deploy_potential(self, iteration: int) -> Path:
        potential_filename = TEMPLATE_POTENTIAL_FILE.format(iteration=iteration)
        deployed_potential = self.potentials_dir / potential_filename

        if self.state_manager.current_potential:
            if self.state_manager.current_potential != deployed_potential:
                shutil.copy(self.state_manager.current_potential, deployed_potential)
        else:
            msg = "No current potential to deploy."
            raise OrchestratorError(msg)

        return deployed_potential

    # --- Phase 1: Zero-Shot Distillation & Baseline Construction ---
    def _phase1_distillation(self) -> None:
        """
        Extracts foundational model capabilities to build a baseline ACE potential.
        """
        if self.state_manager.current_potential and self.state_manager.current_potential.exists():
            self.logger.info(
                f"Phase 1 Skipped: Using existing potential: {self.state_manager.current_potential}"
            )
            return

        self.logger.info("Phase 1: Starting Zero-Shot Distillation.")
        paths = self.dir_manager.setup_iteration(0)

        # 1. & 2. Combinatorial Exploration & DIRECT Sampling (Generator + ActiveSetSelector)
        if not self.generator or not self.oracle or not self.trainer:
            msg = "Required modules for Phase 1 are not initialized."
            raise OrchestratorError(msg)

        candidates_file = paths["candidates"] / FILENAME_CANDIDATES
        training_file = paths["training"] / FILENAME_TRAINING

        # Generate massive pool and sample directly
        try:
            n_candidates = self.config.workflow.distillation.sampling_counts
            candidate_stream = self.generator.generate(n_candidates=n_candidates)

            # In a full implementation, ActiveSetSelector would filter here.
            # For now we write to candidates file
            total_gen = self._stream_write(
                candidate_stream,
                candidates_file,
                batch_size=self.config.workflow.batch_size,
                append=True,
            )
            self.logger.info(f"Phase 1: Generated {total_gen} baseline structures.")
        except Exception as e:
            msg = f"Phase 1 Exploration failed: {e}"
            raise OrchestratorError(msg) from e

        # 3. MACE Confidence Filtering (Oracle processing)
        # Using tiered oracle or purely MACEManager to label and filter based on confidence.
        # The oracle logic (like TieredOracle) automatically assigns MACE uncertainties.
        try:
            cand_stream = iread(str(candidates_file), index=":", format="extxyz")
            labelled_stream = self.oracle.compute(
                cand_stream, batch_size=self.config.workflow.batch_size
            )

            # Filter low uncertainty
            threshold = self.config.workflow.distillation.uncertainty_threshold

            def filter_confident(stream: Iterable[Atoms]) -> Iterable[Atoms]:
                for atoms in stream:
                    if "mace_uncertainty" in atoms.arrays:
                        unc = np.max(atoms.get_array("mace_uncertainty"))  # type: ignore[no-untyped-call]
                        if unc <= threshold:
                            yield atoms
                    else:
                        yield atoms

            confident_stream = filter_confident(labelled_stream)
            total_lbl = self._stream_write(
                confident_stream,
                training_file,
                batch_size=self.config.workflow.batch_size,
                append=True,
            )
            self.logger.info(f"Phase 1: Filtered and retained {total_lbl} confident structures.")
        except Exception as e:
            msg = f"Phase 1 Labeling failed: {e}"
            raise OrchestratorError(msg) from e

        # 4. Baseline ACE Training (LJ Delta Learning)
        # Uses pacemaker with lj baseline defined in strategy
        potential_path = self.trainer.train(
            training_data_path=training_file, initial_potential=None
        )

        if potential_path:
            self.state_manager.current_potential = (
                Path(potential_path) if isinstance(potential_path, str) else potential_path
            )
            self.state_manager.iteration = 0
            self.state_manager.save()
            self.logger.info(
                f"Phase 1 Complete: Initial potential trained: {self.state_manager.current_potential}"
            )
        else:
            msg = "Phase 1: Baseline training failed to produce a potential."
            raise OrchestratorError(msg)

    # --- Phase 2: Validation & Stress Test ---
    def _phase2_validation(self, potential_target: Path) -> bool:
        """
        Verifies minimum physical stability.
        """
        self.logger.info("Phase 2: Validation & Stress Test.")

        # 1. Physical Property Inspection
        if self.validator:
            production_dir = Path(DEFAULT_PRODUCTION_DIR)
            production_dir.mkdir(exist_ok=True)
            report_path = (
                production_dir / f"validation_report_iter_{self.state_manager.iteration}.html"
            )
            structure = self._get_initial_structure(self.state_manager.iteration)

            if structure:
                result = self.validator.validate(potential_target, report_path, structure)
                if not (result.phonon_stable and result.elastic_stable):
                    self.logger.warning(
                        "Phase 2: Validation FAILED. (Auto-fallback to Phase 1 retraining not fully implemented, continuing...)"
                    )
                    return False
                self.logger.info("Phase 2: Validation PASSED.")
            else:
                self.logger.warning("Phase 2: No structure for validation.")
        else:
            self.logger.info("Phase 2: No Validator initialized, skipping.")

        # 2. Miniature MD Stress Test
        # (Implicitly handled as the beginning of Phase 3 if MD fails immediately)
        return True

    # --- Phase 3: Intelligent Cutout & Passivation ---
    def _phase3_intelligent_cutout(self, halt_structure_path: str) -> Atoms | None:
        """
        Extracts valid, clean clusters that DFT can process when MD halts.
        """
        self.logger.info("Phase 3: Intelligent Cutout & Passivation.")
        try:
            halt_structure = read(halt_structure_path)
            if isinstance(halt_structure, list):
                halt_structure = halt_structure[-1]

            # 1. Epicentre Identification
            # (Tiered logic happens during MD, resulting in halt_structure with max_gamma)
            center_idx = self._get_max_gamma_atom_index(halt_structure)

            # 2. Spherical Cutout & Weighting
            # 3. Boundary Pre-relaxation
            # 4. Auto-Passivation
            # All handled securely in `extract_intelligent_cluster`

            cluster = extract_intelligent_cluster(
                halt_structure, [center_idx], self.config.workflow.cutout
            )
        except Exception:
            self.logger.exception("Phase 3: Failed to extract local cluster.")
            return None
        else:
            self.logger.info("Phase 3: Successfully extracted passivated cluster.")
            return cluster

    # --- Phase 4: Hierarchical Fine-Tuning ---
    def _phase4_hierarchical_finetuning(
        self, s0_cluster: Atoms, potential_path: Path, paths: dict[str, Path]
    ) -> Path | None:
        """
        Updates MACE and ACE incrementally, generating surrogate data.
        """
        self.logger.info("Phase 4: Hierarchical Fine-Tuning.")
        if (
            not self.generator
            or not self.active_set_selector
            or not self.oracle
            or not self.trainer
        ):
            self.logger.error("Phase 4: Missing required modules.")
            return None

        # 1. Clean DFT Calculation (Phase 3 transition) & Awaken MACE
        # In a real run, `self.oracle.compute` using TieredOracle handles falling back to DFT
        # and we would fine-tune MACE here if we had `FinetuneManager` hooked up actively.

        # 2. Explosive Surrogate Data Generation
        local_n = self.config.workflow.otf.local_n_candidates
        candidates_gen = self.generator.generate_local(
            s0_cluster, n_candidates=local_n, engine=self.engine, potential=potential_path
        )

        n_select = self.config.workflow.otf.local_n_select
        selected_gen = self.active_set_selector.select(
            candidates_gen, potential_path, n_select=n_select, anchor=s0_cluster
        )

        # Label via Awakened MACE (or Tiered Oracle)
        labelled_gen = self.oracle.compute(selected_gen)

        training_file = paths["training"] / FILENAME_TRAINING
        count = self._stream_write(
            labelled_gen, training_file, batch_size=self.config.workflow.batch_size, append=True
        )
        self.logger.info(f"Phase 4: Added {count} new surrogate structures.")

        # 3. Incremental ACE Update
        # Trainer handles Replay Buffer internally via IncrementalTrainer logic
        new_pot = self.trainer.train(
            training_data_path=training_file, initial_potential=potential_path
        )

        # 4. Seamless Resume happens implicitly by updating current_potential
        # and entering the next loop iteration (which uses Master-Slave Resume in LammpsEngine).

        if new_pot:
            self.logger.info("Phase 4: Fine-Tuning Complete.")
            return Path(new_pot) if isinstance(new_pot, str) else new_pot
        return None

    def _execute_iteration_logic(self, iteration: int, paths: dict[str, Path]) -> None:
        deployed_potential = self._deploy_potential(iteration)

        # Phase 2 Validation (at start of iteration or after training)
        if iteration > 1:
            self._phase2_validation(deployed_potential)

        initial_structure = self._get_initial_structure(iteration)
        if not initial_structure:
            self.logger.warning("No structure for MD. Skipping iteration.")
            return

        # Engine run (Master-Slave Resume handles continuous MD inherently)
        if not self.engine:
            return

        result = self.engine.run(structure=initial_structure, potential=deployed_potential)

        if result and result.halted and result.halt_structure_path:
            threshold_call_dft = self.config.workflow.thresholds.threshold_call_dft
            if result.max_gamma > threshold_call_dft:
                self.logger.info(
                    f"MD Halted at step {result.n_steps} (Gamma {result.max_gamma} > {threshold_call_dft}). Triggering Phase 3."
                )

                # Phase 3
                s0_cluster = self._phase3_intelligent_cutout(result.halt_structure_path)

                if s0_cluster:
                    # Phase 4
                    new_potential = self._phase4_hierarchical_finetuning(
                        s0_cluster, deployed_potential, paths
                    )
                    if new_potential and new_potential.exists():
                        self.state_manager.current_potential = new_potential
                        self.logger.info(f"Potential refined to: {new_potential}")
                    else:
                        self.logger.error("Phase 4 failed to produce a valid potential.")
        elif result:
            self.logger.info(LOG_ITERATION_COMPLETED.format(iteration=iteration))

    def _run_loop_iteration(self) -> None:
        iteration = self.state_manager.iteration + 1
        paths = self.dir_manager.setup_iteration(iteration)
        self.logger.info(
            LOG_START_ITERATION.format(
                iteration=iteration, max_iterations=self.config.workflow.max_iterations
            )
        )

        try:
            self._execute_iteration_logic(iteration, paths)
            self.state_manager.iteration = iteration
            self.state_manager.save()
        except Exception as e:
            self.logger.exception(f"Iteration {iteration} failed")
            msg = f"Iteration {iteration} failed: {e}"
            raise OrchestratorError(msg) from e

    def _finalize(self) -> None:
        production_dir = Path(DEFAULT_PRODUCTION_DIR)
        production_dir.mkdir(exist_ok=True)
        potential_target = production_dir / FILENAME_POTENTIAL

        if self.state_manager.current_potential:
            shutil.copy(self.state_manager.current_potential, potential_target)
            self.logger.info(f"Deployed best potential to {potential_target}")

            self.logger.info("Executing Final Phase 2 Validation.")
            self._phase2_validation(potential_target)

    def run(self) -> None:
        self.logger.info(LOG_START_LOOP)

        try:
            self.initialize_modules()

            # Phase 1
            self._phase1_distillation()

            # MD Loop Phase
            while self.state_manager.iteration < self.config.workflow.max_iterations:
                self._run_loop_iteration()

            self._finalize()
            self.logger.info(LOG_WORKFLOW_COMPLETED)

        except Exception as e:
            self.logger.critical(LOG_WORKFLOW_CRASHED.format(error=e))
            raise
