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
) -> tuple[dict[str, Any] | None, Exception | None]:
    """Top-level helper to run a single calculation attempt. Returns serialized results and any exception for ProcessPoolExecutor."""
    try:
        # Create new calculator for clean state
        # Use provided temporary directory to prevent file collisions and race conditions
        calc = driver.get_calculator(atoms, config.model_copy(), directory=calc_dir)
        atoms.calc = calc

        # Trigger actual calculation
        energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        forces = atoms.get_forces()  # type: ignore[no-untyped-call]

        # Try to get stress (optional)
        stress = None
        with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
            stress = atoms.get_stress()  # type: ignore[no-untyped-call]

        results = {"energy": energy, "forces": forces, "stress": stress}

    except Exception as e:
        return None, e
    else:
        return results, None


class DFTManager(BaseOracle):
    """
    Manages DFT calculations with self-healing capabilities.

    Memory Usage:
        This class processes structures one-by-one (streaming) to ensure O(1) memory usage
        relative to the dataset size. It does not materialize the input iterator into a list.
    """

    def __init__(self, config: DFTConfig, driver: Any | None = None) -> None:
        """
        Initializes the DFTManager.

        Args:
            config: DFT configuration.
            driver: Optional Driver instance (for dependency injection).
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
        for _i, atoms in enumerate(structures):
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as work_dir:
                work_path = Path(work_dir)
                yield self._process_structure(atoms, str(work_path))

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

            # Create a completely clean isolated subdirectory for each attempt to prevent cross-process data leakage
            attempt_dir = Path(calc_dir) / f"attempt_{i}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Architecture: Add explicit execution timeout for DFT manager to prevent hangs
                import concurrent.futures

                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _run_calculator_process,
                        self.driver,
                        atoms,
                        current_config,
                        str(attempt_dir),
                    )
                    # Set a hard limit of 3600 seconds per self-healing attempt
                    results, exception = future.result(timeout=3600)

                    if exception:
                        self._handle_exception(exception)

                    # Apply results from subprocess back to the atoms object in main process
                    # Architecture: We use SinglePointCalculator to avoid sharing stateful calculators across process boundaries
                    from ase.calculators.singlepoint import SinglePointCalculator

                    if results:
                        calc_kwargs = {"energy": results["energy"], "forces": results["forces"]}
                        if results.get("stress") is not None:
                            calc_kwargs["stress"] = results["stress"]

                        calc = SinglePointCalculator(atoms, **calc_kwargs)  # type: ignore[no-untyped-call]
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


from typing import Protocol

class ModelLoaderProtocol(Protocol):
    def load(self, model_path: str) -> Any:
        ...

class DefaultMACELoader:
    def load(self, model_path: str) -> Any:
        import os
        from pyacemaker.domain_models.defaults import DEFAULT_POTENTIALS_DIR

        # Canonicalize the path using os.path.realpath to safely unpack symlinks and avoid TOCTOU
        canonical_path_str = os.path.realpath(model_path)
        canonical_path = Path(canonical_path_str)

        # Verify containment: ensure the path falls inside the accepted allowed_base_dir.
        allowed_dir = Path(DEFAULT_POTENTIALS_DIR).resolve()

        # Security: Proceed with strict canonical path comparison using relative_to
        try:
            canonical_path.relative_to(allowed_dir)
        except ValueError as e:
            msg = f"MACE model path {canonical_path} is outside allowed directory {allowed_dir}"
            raise ValueError(msg) from e

        if not canonical_path.exists():
            msg = f"MACE model path does not exist: {canonical_path}"
            raise FileNotFoundError(msg)
        if not canonical_path.is_file():
            msg = f"MACE model path must be a file: {canonical_path}"
            raise ValueError(msg)

        return str(canonical_path)


class MACEManager(BaseOracle):
    """
    Wrapper for MACE foundation model inferences.
    Provides energy, forces, and uncertainty estimation.
    """

    def __init__(self, model_path: str, loader: ModelLoaderProtocol | None = None) -> None:
        self.loader = loader or DefaultMACELoader()
        self.model_path = self.loader.load(model_path)
        self.is_initialized = True

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        for atoms in structures:
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
        primary_oracle: BaseOracle,
        fallback_oracle: BaseOracle,
        thresholds: ActiveLearningThresholds,
    ) -> None:
        if primary_oracle is None:
            msg = "Primary oracle must be provided."
            raise ValueError(msg)

        if fallback_oracle is None:
            msg = "Fallback oracle cannot be None."
            raise ValueError(msg)

        self.primary = primary_oracle
        self.fallback = fallback_oracle
        self.thresholds = thresholds

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        for atoms in structures:
            # First query Primary Oracle (e.g. MACE)
            primary_result = next(self.primary.compute(iter([atoms])))

            # Evaluate uncertainty if available
            try:
                c_gamma = primary_result.get_array("c_gamma")  # type: ignore[no-untyped-call]
                max_uncertainty = np.max(c_gamma)
            except KeyError:
                # If the primary oracle doesn't provide uncertainty, assume it's confident
                # or defer entirely to fallback strategy. For now, assume safe.
                max_uncertainty = 0.0
                c_gamma = None

            if max_uncertainty > self.thresholds.threshold_call_dft:
                # Fallback to Secondary Oracle (e.g. DFT)
                logger.info(
                    f"Uncertainty {max_uncertainty:.4f} > {self.thresholds.threshold_call_dft}. Falling back to secondary oracle."
                )

                # Evaluate the whole structure as per fallback logic.
                fallback_result = next(self.fallback.compute(iter([atoms])))

                # We should retain the c_gamma array for active learning tracking
                if c_gamma is not None:
                    fallback_result.set_array("c_gamma", c_gamma)  # type: ignore[no-untyped-call]

                yield fallback_result
            else:
                yield primary_result
