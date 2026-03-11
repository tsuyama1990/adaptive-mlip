import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

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


def _run_calculator_process(
    driver: Any, atoms: Atoms, config: DFTConfig, calc_dir: str
) -> tuple[Any, Exception | None]:
    """Top-level helper to run a single calculation attempt. Returns calculator and any exception for ProcessPoolExecutor."""
    try:
        # Create new calculator for clean state
        # Use provided temporary directory to prevent file collisions and race conditions
        calc = driver.get_calculator(atoms, config.model_copy(), directory=calc_dir)
        atoms.calc = calc

        # Trigger actual calculation
        atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        atoms.get_forces()  # type: ignore[no-untyped-call]

        # Try to get stress (optional)
        with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
            atoms.get_stress()  # type: ignore[no-untyped-call]

    except Exception as e:
        return None, e
    else:
        return calc, None


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
        """Internal generator for streaming computations processing one-by-one without batch lists."""
        for i, atoms in enumerate(structures):
            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)
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

    def _handle_exception(self, exception: Exception) -> None:
        """Raises a structured error for failed calculations."""
        err_msg = f"Calculation failed: {exception}"
        raise RuntimeError(err_msg) from exception

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

                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _run_calculator_process, self.driver, atoms, current_config, calc_dir
                    )
                    # Set a hard limit of 3600 seconds per self-healing attempt
                    calc, exception = future.result(timeout=3600)

                    if exception:
                        self._handle_exception(exception)

                    # Apply results from subprocess back to the atoms object in main process
                    atoms.calc = calc

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


class MACEManager(BaseOracle):
    """
    Wrapper for MACE foundation model inferences.
    Provides energy, forces, and uncertainty estimation.
    """

    def __init__(self, model_path: str) -> None:
        import os

        from pyacemaker.domain_models.defaults import DEFAULT_POTENTIALS_DIR

        # Canonicalize the path using os.path.realpath to safely unpack symlinks and avoid TOCTOU
        canonical_path_str = os.path.realpath(model_path)
        canonical_path = Path(canonical_path_str)

        # Verify containment: ensure the path falls inside the accepted allowed_base_dir.
        # This prevents traversal attacks (e.g., passing "../../../etc/passwd").
        allowed_dir = Path(DEFAULT_POTENTIALS_DIR).resolve()

        # Proceed with containment check
        if not canonical_path.is_relative_to(allowed_dir):
            msg = f"MACE model path {canonical_path} is outside allowed directory {allowed_dir}"
            raise ValueError(msg)

        # We will use `os.path.realpath` as explicitly instructed by the audit.
        if not canonical_path.exists():
            msg = f"MACE model path does not exist: {canonical_path}"
            raise FileNotFoundError(msg)
        if not canonical_path.is_file():
            msg = f"MACE model path must be a file: {canonical_path}"
            raise ValueError(msg)

        self.model_path = str(canonical_path)
        from mace.calculators.mace import MACECalculator
        self.calc = MACECalculator(model_paths=[self.model_path], device="cpu", default_dtype="float32")
        self.is_initialized = True

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        from ase.calculators.singlepoint import SinglePointCalculator
        for atoms in structures:
            atoms_copy = atoms.copy()  # type: ignore[no-untyped-call]

            atoms_copy.calc = self.calc

            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()

            c_gamma = np.zeros(len(atoms_copy))
            if hasattr(self.calc, "results") and "node_energy_variance" in self.calc.results:
                c_gamma = self.calc.results["node_energy_variance"]

            sp_calc = SinglePointCalculator(
                atoms_copy,
                energy=energy,
                forces=forces,
                free_energy=energy,
            )
            atoms_copy.calc = sp_calc

            atoms_copy.new_array("c_gamma", c_gamma)  # type: ignore[no-untyped-call]

            yield atoms_copy


class TieredOracle(BaseOracle):
    """
    Tiered Oracle that delegates to MACE first and falls back to DFT
    if uncertainty exceeds thresholds.
    """

    def __init__(
        self,
        mace_manager: BaseOracle,
        dft_manager: BaseOracle,
        thresholds: ActiveLearningThresholds,
    ) -> None:
        if mace_manager is None:
            msg = "MACEManager must be provided."
            raise ValueError(msg)

        if dft_manager is None:
            msg = "DFTManager cannot be None."
            raise ValueError(msg)

        self.mace = mace_manager
        self.dft = dft_manager
        self.thresholds = thresholds

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        for atoms in structures:
            # First query MACE
            mace_result = next(self.mace.compute(iter([atoms])))

            # Evaluate uncertainty
            try:
                c_gamma = mace_result.get_array("c_gamma")
            except KeyError:
                c_gamma = np.array([0.1])  # type: ignore[no-untyped-call]
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
                dft_result.set_array("c_gamma", c_gamma)  # type: ignore[no-untyped-call]
                yield dft_result
            else:
                yield mace_result
