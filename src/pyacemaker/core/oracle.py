import contextlib
import logging
import tempfile
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.healing import (
    HealingStrategy,
    IncreaseSmearingStrategy,
    ReduceBetaStrategy,
    UseCGDiagonalizationStrategy,
)
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import (
    ERR_ORACLE_FAILED,
    ERR_ORACLE_ITERATOR,
    ERR_ORACLE_WARN_EMPTY,
)
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

logger = logging.getLogger(__name__)

class RetryManager:
    """Handles the self-healing retry loop."""
    def __init__(self, strategies: list[HealingStrategy | None]) -> None:
        self.strategies = strategies

    def execute(self, atoms: Atoms, base_config: DFTConfig, calc_dir: str, runner: Callable[[Atoms, DFTConfig, str], None]) -> Atoms:
        current_config = base_config.model_copy()
        last_error: Exception | None = None

        for i, strategy in enumerate(self.strategies):
            if strategy:
                strategy.apply(current_config)
                strategy_name = strategy.__class__.__name__
            else:
                strategy_name = "Initial"

            try:
                runner(atoms, current_config, calc_dir)
            except (RuntimeError, ValueError) as e:
                last_error = e
                atoms.calc = None  # Clean up failed calculator
                logger.warning(
                    f"DFT calculation attempt {i+1} ({strategy_name}) failed. Error: {e!s}. Retrying..."
                )
                continue
            else:
                return atoms

        raise OracleError(ERR_ORACLE_FAILED.format(error=last_error)) from last_error


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

        self.retry_manager = RetryManager([
            None,
            ReduceBetaStrategy(config.mixing_beta_factor),
            IncreaseSmearingStrategy(config.smearing_width_factor),
            UseCGDiagonalizationStrategy()
        ])

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

        count = 0
        iterator_exhausted = False

        while not iterator_exhausted:
            # We process structures one-by-one without materializing a list,
            # using a temporary directory for a batch of up to `batch_size` items.
            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)
                for i in range(batch_size):
                    try:
                        atoms = next(structures)
                    except StopIteration:
                        iterator_exhausted = True
                        break

                    calc_dir = work_path / f"calc_{i}"
                    calc_dir.mkdir()

                    # Yielding inside a tempdir context can be tricky if the consumer stops iteration early.
                    # We process the structure, store the result, and then yield it so the generator
                    # flow controls the tempo and resources are cleaned up correctly even if interrupted.
                    processed_atoms = self._process_structure(atoms, str(calc_dir))
                    yield processed_atoms
                    count += 1

        if count == 0:
            warnings.warn(ERR_ORACLE_WARN_EMPTY, stacklevel=2)

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

        # Validate embedded cluster size to prevent massive DFT calculations
        if len(structure_to_compute) > 10000:
            msg = f"Embedded structure size ({len(structure_to_compute)}) exceeds safe computational limits (10000 atoms)."
            raise ValueError(msg)

        return self._compute_single(structure_to_compute, calc_dir)

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
        return self.retry_manager.execute(atoms, self.config, calc_dir, self._run_calculator)

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
