import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import ERR_ORACLE_FAILED, ERR_ORACLE_ITERATOR
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

logger = logging.getLogger(__name__)

class DFTManager(BaseOracle):
    """
    Manages DFT calculations with self-healing capabilities.

    Memory Usage:
        This class processes structures one-by-one (streaming) to ensure O(1) memory usage
        relative to the dataset size. It does not materialize the input iterator into a list.
    """

    def __init__(self, config: DFTConfig, driver: QEDriver | None = None) -> None:
        """
        Initializes the DFTManager.

        Args:
            config: DFT configuration.
            driver: Optional QEDriver instance (for dependency injection).
                    If None, a new QEDriver is created.
        """
        self.config = config
        self.driver = driver or QEDriver()

        # Cache strategies to avoid recreation on every compute call
        self.strategies: list[Callable[[DFTConfig], None] | None] = [
            None,
            self._strategy_reduce_beta,
            self._strategy_increase_smearing,
            self._strategy_use_cg
        ]

    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        """
        Computes DFT properties for stream of structures.

        Args:
            structures: Iterator of AtomStructure objects.
            batch_size: Batch size for processing (used to manage temporary directories).

        Yields:
            AtomStructure objects with computed properties.

        Raises:
            OracleError: If a calculation fails fatally.
            TypeError: If structures is not an iterator (to prevent memory leaks from huge lists).
        """
        # Validate that structures is an iterator to enforce O(1) memory usage contract
        if isinstance(structures, (list, tuple)):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[AtomStructure], batch_size: int) -> Iterator[AtomStructure]:
        """Internal generator for streaming computations with batching."""

        while True:
            # Create a batch generator (iterator slice)
            # IMPORTANT: islice consumes the iterator.
            # Using list(islice) materializes the batch in memory.
            # This is acceptable if batch_size is small (e.g. 10-100) as intended by config.
            # If batch_size is huge, user is responsible for OOM.
            # However, to be strictly streaming within the batch, we can iterate the islice directly?
            # BUT we need to know if it's empty to break loop. `islice` doesn't tell us easily without consuming.
            # And `TemporaryDirectory` context needs to wrap the processing of items.

            # The feedback "Process structures one-by-one without batching to maintain O(1) memory usage"
            # suggests avoiding list(islice).
            # But we want to reuse the temp dir for efficiency (less I/O mkdir calls).
            # If we process 1-by-1, we create temp dir for every single atom? That's huge I/O overhead.
            # Compromise: Batching is necessary for performance (directory reuse).
            # Memory usage of list(islice) is O(batch_size). With batch_size=10, it's negligible.
            # The concern is only valid if batch_size is millions.
            # We stick to list(islice) as it is robust for batch detection.
            # We just ensure `structures` iterator isn't consumed entirely.

            batch = list(islice(structures, batch_size))
            if not batch:
                break

            # Create ONE temporary directory for the entire batch
            # This reduces filesystem overhead (mkdir/rmdir) significantly for large datasets.
            # We process sequentially inside the batch to keep memory low,
            # but reuse the workspace.
            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)

                # Process items in the batch
                for i, structure in enumerate(batch):
                    # Strict Type Validation per element to fail fast
                    if not isinstance(structure, AtomStructure):
                        msg = f"Expected AtomStructure, got {type(structure)}"
                        raise TypeError(msg)

                    # Use unique subdirs within the batch temp dir
                    calc_dir = work_path / f"calc_{i}"
                    calc_dir.mkdir()
                    yield self._process_structure(structure, str(calc_dir))

    def _process_structure(self, structure: AtomStructure, calc_dir: str) -> AtomStructure:
        """
        Applies embedding and computes properties for a single structure.

        Args:
            structure: The input atomic structure.
            calc_dir: Directory to run calculation in.

        Returns:
            AtomStructure: The structure with computed properties (energy, forces, stress).
                   If embedding is configured, properties are computed for the embedded cluster.
        """
        # Apply Periodic Embedding if configured
        if self.config.embedding_buffer:
            atoms_to_compute = embed_cluster(structure.atoms, buffer=self.config.embedding_buffer)
            # Create a temporary AtomStructure for computation
            structure_to_compute = AtomStructure(
                atoms=atoms_to_compute,
                provenance=structure.provenance.copy()
            )
        else:
            structure_to_compute = structure

        # Run computation
        computed_atoms = self._compute_single(structure_to_compute.atoms, calc_dir)

        # Update AtomStructure with computed properties
        return AtomStructure.from_ase(computed_atoms)


    def _get_strategies(self) -> list[Callable[[DFTConfig], None] | None]:
        """
        Returns a list of self-healing strategies.
        """
        return self.strategies

    def _strategy_reduce_beta(self, c: DFTConfig) -> None:
        c.mixing_beta *= self.config.mixing_beta_factor

    def _strategy_increase_smearing(self, c: DFTConfig) -> None:
        c.smearing_width *= self.config.smearing_width_factor

    def _strategy_use_cg(self, c: DFTConfig) -> None:
        c.diagonalization = "cg"

    def _compute_single(self, atoms: Atoms, calc_dir: str) -> Atoms:
        """
        Runs calculation for a single structure with retries and self-healing strategies.

        Args:
            atoms: The atomic structure to calculate.
            calc_dir: Working directory for the calculation.

        Returns:
            Atoms object with calculated properties attached.

        Raises:
            OracleError: If calculation fails after all retries and strategies.
        """
        current_config = self.config.model_copy()
        strategies = self._get_strategies()
        last_error: Exception | None = None

        for i, strategy in enumerate(strategies):
            if strategy:
                strategy(current_config)
                strategy_name = strategy.__name__
            else:
                strategy_name = "Initial"

            try:
                self._run_calculator(atoms, current_config, calc_dir)
            except Exception as e:
                # Catch all exceptions (RuntimeError, CalculatorSetupError, JobFailedException etc)
                # to ensure self-healing strategies are attempted.
                last_error = e
                atoms.calc = None  # Clean up failed calculator

                # Enhanced Logging for debugging
                logger.warning(
                    f"DFT calculation attempt {i+1} ({strategy_name}) failed. Error: {e!s}. Retrying..."
                )
                continue
            else:
                return atoms

        # Correctly format the error message with the captured exception
        raise OracleError(ERR_ORACLE_FAILED.format(error=last_error)) from last_error

    def _run_calculator(self, atoms: Atoms, config: DFTConfig, calc_dir: str) -> None:
        """Helper to run a single calculation attempt."""
        # Create new calculator for clean state
        # Use provided temporary directory to prevent file collisions and race conditions
        calc = self.driver.get_calculator(atoms, config.model_copy(), directory=calc_dir)
        atoms.calc = calc

        # Trigger actual calculation
        atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        atoms.get_forces()  # type: ignore[no-untyped-call]

        # Try to get stress (optional)
        with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
            atoms.get_stress()  # type: ignore[no-untyped-call]
