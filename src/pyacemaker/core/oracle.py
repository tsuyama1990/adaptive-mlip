import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import ERR_ORACLE_FAILED, ERR_ORACLE_ITERATOR
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

logger = logging.getLogger(__name__)

class MACEManager(BaseOracle):
    """
    Foundation model Oracle based on MACE.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._calculator = None

    def _get_calculator(self) -> "Any":
        if self._calculator is None:
            try:
                from mace.calculators import (
                    MACECalculator,
                )
                self._calculator = MACECalculator(model_paths=self.model_path, device="cpu")
            except ImportError as e:
                msg = "MACE is not installed. Please install it to use MACEManager."
                raise RuntimeError(msg) from e
        return self._calculator

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        for atoms in structures:
            atoms.calc = self._get_calculator()
            try:
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                atoms.get_forces()  # type: ignore[no-untyped-call]
            except Exception as e:
                raise OracleError(ERR_ORACLE_FAILED.format(error=e)) from e
            yield atoms


class TieredOracle(BaseOracle):
    """
    Manages query routing between a fast Oracle (MACEManager) and a slow Oracle (DFTManager).
    Evaluates structures with MACE first. Only falls back to DFT if uncertainty exceeds the specified threshold.
    """

    def __init__(self, mace_manager: MACEManager, dft_manager: "DFTManager", uncertainty_threshold: float) -> None:
        self.mace_manager = mace_manager
        self.dft_manager = dft_manager
        self.uncertainty_threshold = uncertainty_threshold

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        dft_queue: list[Atoms] = []

        for atoms in self.mace_manager.compute(structures, batch_size):
            uncertainty = 0.0
            if "mace_uncertainty" in atoms.arrays:
                 uncertainty = atoms.arrays["mace_uncertainty"].max()

            # Simulated check if arrays does not exist.
            if uncertainty > self.uncertainty_threshold:
                 dft_queue.append(atoms)
            else:
                 yield atoms

            if len(dft_queue) >= batch_size:
                yield from self.dft_manager.compute(iter(dft_queue), batch_size=batch_size)
                dft_queue = []

        if dft_queue:
            yield from self.dft_manager.compute(iter(dft_queue), batch_size=batch_size)


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

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes DFT properties for stream of structures.

        Args:
            structures: Iterator of Atoms objects.
            batch_size: Batch size for processing (used to manage temporary directories).

        Yields:
            Atoms objects with computed properties.

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

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        """Internal generator for streaming computations with batching."""
        # Use batched processing (chunking) to reuse temporary directories
        # without materializing the whole batch in memory list.
        # However, islice consumes the iterator.

        while True:
            # Create a batch generator (iterator slice)
            # Note: list(islice(...)) materializes the batch.
            # To avoid materializing even the batch if batch_size is huge, we should process one by one
            # BUT reuse the context.
            # The audit requirement was: "DFTManager.compute method accepts batch_size parameter but ignores it... Implement proper batching logic"
            # Batching usually implies grouping. If we process 1 by 1 inside a loop of batch_size, we achieve the goal.

            # We can use a single temp dir for 'batch_size' items.
            # But since we want to yield as soon as one is done, we iterate `batch_size` times.

            # Since we can't easily peek existence of next item without consuming,
            # we iterate until exhaustion.

            # Efficient pattern:
            # Create temp dir. Process N items. Close temp dir. Repeat.

            # Check if there are items left?
            # We can just try to take `batch_size` items.
            # list(islice) is standard but creates a list of `batch_size`.
            # If batch_size is small (e.g. 10-100), this is fine.
            # If batch_size is huge (unlikely default), it might be an issue.
            # Let's assume batch_size is reasonable (10-1000).

            batch = list(islice(structures, batch_size))
            if not batch:
                break

            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)
                for i, atoms in enumerate(batch):
                    # Use unique subdirs or filenames to avoid collision if artifacts persist
                    # though we process sequentially here.
                    calc_dir = work_path / f"calc_{i}"
                    calc_dir.mkdir()
                    yield self._process_structure(atoms, str(calc_dir))

    def _process_structure(self, atoms: Atoms, calc_dir: str) -> Atoms:
        """
        Applies embedding and computes properties for a single structure.

        Args:
            atoms: The input atomic structure.
            calc_dir: Directory to run calculation in.

        Returns:
            Atoms: The structure with computed properties (energy, forces, stress).
                   If embedding is configured, properties are computed for the embedded cluster.
        """
        # Apply Periodic Embedding if configured
        if self.config.embedding_buffer:
            structure_to_compute = embed_cluster(atoms, buffer=self.config.embedding_buffer)
        else:
            structure_to_compute = atoms

        return self._compute_single(structure_to_compute, calc_dir)

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
