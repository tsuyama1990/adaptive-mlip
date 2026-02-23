import contextlib
import logging
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
    LOG_MD_COMPLETED,
    TEMPLATE_POTENTIAL_FILE,
)
from pyacemaker.domain_models.md import MDSimulationResult
from pyacemaker.utils.misc import batched

logger = logging.getLogger(__name__)


class OTFManager:
    """
    Manages the On-The-Fly (OTF) Active Learning loop.
    Encapsulates logic for running MD, monitoring uncertainty, and triggering retraining.
    """

    def __init__(
        self,
        config: PyAceConfig,
        generator: BaseGenerator,
        oracle: BaseOracle,
        trainer: BaseTrainer,
        engine: BaseEngine,
        active_set_selector: ActiveSetSelector,
    ) -> None:
        self.config = config
        self.generator = generator
        self.oracle = oracle
        self.trainer = trainer
        self.engine = engine
        self.active_set_selector = active_set_selector

    def run_loop(
        self,
        paths: dict[str, Path],
        potential_path: Path | None,
        iteration: int,
        potentials_dir: Path,
    ) -> None:
        """
        Executes the OTF loop: Deploy -> Run MD -> Check -> Fine-tune -> Repeat.

        Args:
            paths: Dictionary of directory paths for the current iteration.
            potential_path: Path to the currently trained potential.
            iteration: Current iteration number (for naming).
            potentials_dir: Directory where potentials are stored/deployed.
        """
        if not potential_path or not potential_path.exists():
            logger.warning("No potential to deploy/run MD.")
            return

        # Deploy potential
        filename = TEMPLATE_POTENTIAL_FILE.format(iteration=iteration)
        target_path = potentials_dir / filename
        with contextlib.suppress(shutil.SameFileError):
            shutil.copy(potential_path, target_path)
        logger.info(f"Deployed potential to {target_path}")

        # Load initial structure for MD
        # Try to use first candidate from current iteration
        candidates_file = paths["candidates"] / FILENAME_CANDIDATES
        initial_structure = None
        try:
            if candidates_file.exists():
                # Use next() on iterator to get just one structure efficiently
                # index=None implies standard iteration
                initial_structure = next(iread(str(candidates_file), index=None))
        except (StopIteration, Exception):
            logger.warning("Could not load initial structure from candidates.")

        if initial_structure is None:
            logger.warning("No valid initial structure for MD. Skipping simulation.")
            return

        current_potential = potential_path
        retries = 0
        max_retries = self.config.workflow.otf.max_retries

        while retries <= max_retries:
            logger.info(f"Starting MD run (attempt {retries}/{max_retries})")

            try:
                result = self.engine.run(structure=initial_structure, potential=current_potential)
            except Exception:
                logger.exception("MD execution failed")
                break

            if not result.halted:
                logger.info(LOG_MD_COMPLETED)
                break

            logger.warning(
                f"MD halted at step {result.n_steps} (Gamma: {result.max_gamma:.2f} > "
                f"{self.config.workflow.otf.uncertainty_threshold})"
            )

            if retries >= max_retries:
                logger.error("Max OTF retries reached. Moving to next iteration.")
                break

            # Handle Halt
            new_potential = self._handle_halt_event(result, current_potential, paths)

            if not new_potential:
                logger.warning("Refinement failed. Aborting OTF loop.")
                break

            current_potential = new_potential

            # Update initial structure for next run to continue from halt
            if result.halt_structure_path:
                try:
                    new_init = read(result.halt_structure_path)
                    if isinstance(new_init, list):
                        new_init = new_init[-1]
                    if isinstance(new_init, Atoms):
                        initial_structure = new_init
                    else:
                        logger.error("Resume structure is invalid.")
                        break
                except Exception:
                    logger.exception("Failed to read halt structure for resume")
                    break

            retries += 1

    def _handle_halt_event(
        self,
        result: MDSimulationResult,
        potential_path: Path,
        paths: dict[str, Path]
    ) -> Path | None:
        """
        Handles an MD halt event by generating local candidates, selecting active set,
        computing properties, and fine-tuning the potential.
        """
        if not result.halt_structure_path:
            logger.error("No halt structure path in result.")
            return None

        try:
            halt_structure = read(result.halt_structure_path)
            if isinstance(halt_structure, list):
                halt_structure = halt_structure[-1]
            if not isinstance(halt_structure, Atoms):
                logger.error("Halt structure is not an Atoms object.")
                return None
        except Exception:
            logger.exception("Failed to read halt structure")
            return None

        # Generate Local Candidates
        local_n = self.config.workflow.otf.local_n_candidates
        candidates_gen = self.generator.generate_local(halt_structure, n_candidates=local_n)

        # Select Active Set
        n_select = self.config.workflow.otf.local_n_select
        try:
            selected_gen = self.active_set_selector.select(
                candidates_gen,
                potential_path,
                n_select=n_select
            )
        except Exception:
            logger.exception("Active set selection failed")
            return None

        # Compute properties (Oracle)
        labelled_gen = self.oracle.compute(selected_gen)

        # Append to training data
        training_file = paths["training"] / FILENAME_TRAINING
        count = 0
        batch_size = self.config.workflow.batch_size

        # Use streaming write with appending
        try:
            with training_file.open("a") as f:
                for batch in batched(labelled_gen, n=batch_size):
                    write(f, list(batch), format="extxyz")
                    count += len(batch)
        except Exception:
            logger.exception("Failed to write labelled data during OTF loop")
            return None

        if count == 0:
            logger.warning("No new training data generated from halt event.")
            return None

        logger.info(f"Added {count} new structures from OTF loop.")

        # Fine-tune the potential
        try:
            new_potential = self.trainer.train(training_file, initial_potential=potential_path)
            return Path(new_potential) if isinstance(new_potential, (str, Path)) else None
        except Exception:
            logger.exception("Fine-tuning failed")
            return None
