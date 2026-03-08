import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import ERR_ORACLE_FAILED, ERR_ORACLE_ITERATOR
from pyacemaker.domain_models.workflow import ActiveLearningThresholds
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
            self._strategy_use_cg,
        ]

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes DFT properties for stream of structures.
        """
        if isinstance(structures, (list, tuple)):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        """Internal generator for streaming computations with batching."""
        while True:
            # Process in batches, yielding immediately, but reusing context
            batch = list(islice(structures, batch_size))
            if not batch:
                break

            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)
                for i, atoms in enumerate(batch):
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
                # Architecture: Add explicit execution timeout for DFT manager to prevent hangs
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._run_calculator, atoms, current_config, calc_dir)
                    # Set a hard limit of 3600 seconds per self-healing attempt
                    future.result(timeout=3600)
            except concurrent.futures.TimeoutError as e:
                last_error = e
                atoms.calc = None
                logger.exception(
                    f"DFT calculation attempt {i + 1} ({strategy_name}) timed out after 3600s. Retrying..."
                )
                continue
            except Exception as e:
                # Catch all exceptions (RuntimeError, CalculatorSetupError, JobFailedException etc)
                # to ensure self-healing strategies are attempted.
                last_error = e
                atoms.calc = None  # Clean up failed calculator

                # Enhanced Logging for debugging
                logger.warning(
                    f"DFT calculation attempt {i + 1} ({strategy_name}) failed. Error: {e!s}. Retrying..."
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


class MACEManager(BaseOracle):
    """
    Wrapper for MACE foundation model inferences.
    Provides energy, forces, and uncertainty estimation.
    """

    def __init__(self, model_path: str = "mace-mp-0-medium") -> None:
        self.model_path = model_path
        # Mock MACE initialization
        self.is_initialized = True

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        while True:
            batch = list(islice(structures, batch_size))
            if not batch:
                break
            for atoms in batch:
                atoms_copy = atoms.copy()  # type: ignore[no-untyped-call]

                # Mock MACE predictions
                energy = -10.0 * len(atoms_copy)
                forces = np.zeros((len(atoms_copy), 3))

                # Mock uncertainty in c_gamma array
                c_gamma = np.random.uniform(0.01, 0.1, size=len(atoms_copy))

                # In a real implementation we would attach a calculator
                # Here we just mock setting the arrays and attributes
                atoms_copy.calc = None
                atoms_copy.info["energy"] = energy
                atoms_copy.new_array("forces", forces)
                atoms_copy.new_array("c_gamma", c_gamma)

                yield atoms_copy


class TieredOracle(BaseOracle):
    """
    Tiered Oracle that delegates to MACE first and falls back to DFT
    if uncertainty exceeds thresholds.
    """

    def __init__(
        self,
        mace_manager: MACEManager,
        dft_manager: DFTManager,
        thresholds: ActiveLearningThresholds,
    ) -> None:
        self.mace = mace_manager
        self.dft = dft_manager
        self.thresholds = thresholds

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        while True:
            batch = list(islice(structures, batch_size))
            if not batch:
                break
            for atoms in batch:
                # First query MACE
                mace_result = next(self.mace.compute(iter([atoms])))

                # Evaluate uncertainty
                c_gamma = mace_result.get_array("c_gamma") # type: ignore[no-untyped-call]
                max_uncertainty = np.max(c_gamma)

                if max_uncertainty > self.thresholds.threshold_call_dft:
                    # Fallback to DFT
                    logger.info(
                        f"Uncertainty {max_uncertainty:.4f} > {self.thresholds.threshold_call_dft}. Falling back to DFT."
                    )

                    # Only pass the atoms exceeding the add_train threshold to DFT?
                    # For now, evaluate the whole structure as per fallback logic.
                    dft_result = next(self.dft.compute(iter([atoms])))

                    # We should retain the c_gamma array for active learning tracking
                    dft_result.set_array("c_gamma", c_gamma) # type: ignore[no-untyped-call]
                    yield dft_result
                else:
                    yield mace_result
