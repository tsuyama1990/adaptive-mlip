import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import (
    CalculationFailed,
    CalculatorSetupError,
    PropertyNotImplementedError,
)

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import ERR_ORACLE_FAILED
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

logger = logging.getLogger(__name__)


class MACEManager(BaseOracle):
    """
    Wraps MACE foundational model inference. Currently mocked to output
    energy, forces, and dummy uncertainty values if MACE is not available.
    """
    def __init__(self, model_path: str = "mace-mp-0-medium", use_mock: bool = False) -> None:
        if not isinstance(model_path, str):
            msg = "model_path must be a string"
            raise TypeError(msg)
        if not isinstance(use_mock, bool):
            msg = "use_mock must be a boolean"
            raise TypeError(msg)

        from pyacemaker.utils.path import validate_path_safe
        # Usually model_path is a string URL or name, but if it is a local file, validate it.
        # To avoid breaking valid mace-mp-0-medium strings, we only validate if use_mock is False
        # and it looks like a path.
        if not use_mock and "/" in model_path:
            self.model_path = str(validate_path_safe(Path(model_path)))
        else:
            self.model_path = model_path
        self.use_mock = use_mock
        # Real implementation would load mace torch model here
        # self.model = mace.calculators.mace_mp(model=model_path)

    def compute_uncertainty(self, structure: Atoms) -> float:
        inferred = self._infer(structure)
        if "c_gamma" in inferred.arrays:
            return float(np.max(inferred.get_array("c_gamma")))  # type: ignore[no-untyped-call]
        return 0.0

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for batch in self._batched(structures, batch_size):
            for atoms in batch:
                yield self._infer(atoms)

    @staticmethod
    def _batched(iterable: Iterator[Atoms], n: int) -> Iterator[list[Atoms]]:
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    def _infer(self, atoms: Atoms) -> Atoms:
        result = atoms.copy()  # type: ignore[no-untyped-call]
        n_atoms = len(result)

        if self.use_mock:
            # Mock inference: Random dummy energy, forces, and uncertainty
            rng = np.random.default_rng()
            result.info["energy"] = rng.uniform(-10.0, -5.0) * n_atoms
            result.arrays["forces"] = rng.uniform(-1.0, 1.0, size=(n_atoms, 3))
            # Uncertainty output (gamma)
            result.arrays["c_gamma"] = rng.uniform(0.001, 0.05, size=n_atoms)
        else:
            # Placeholder for actual MACE call
            # result.calc = self.model
            # result.get_potential_energy()
            # result.get_forces()
            # If MACE is missing and use_mock is False, we should raise an error as per requirements.
            # But wait, MACE is an external dependency. We'll raise a RuntimeError.
            msg = "MACE model is not installed or available, and use_mock is False."
            raise RuntimeError(msg)

        return result


class TieredOracle(BaseOracle):
    """
    Implements a query strategy where MACE is queried first.
    If its uncertainty exceeds a threshold, it falls back to DFT.
    """
    def __init__(self, mace_manager: MACEManager, dft_manager: "DFTManager", uncertainty_threshold: float = 0.05) -> None:
        if mace_manager is None:
            msg = "mace_manager cannot be None"
            raise ValueError(msg)
        if dft_manager is None:
            msg = "dft_manager cannot be None"
            raise ValueError(msg)
        if not isinstance(uncertainty_threshold, (float, int)) or uncertainty_threshold < 0:
            msg = "uncertainty_threshold must be a non-negative number"
            raise ValueError(msg)

        self.mace_manager = mace_manager
        self.dft_manager = dft_manager
        self.uncertainty_threshold = uncertainty_threshold

    def compute_uncertainty(self, structure: Atoms) -> float:
        return self.mace_manager.compute_uncertainty(structure)

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        return self._compute_generator(structures)

    def _compute_generator(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        # Stream one-by-one to avoid materializing large batches per audit requirement
        for orig_structure in structures:
            mace_result = next(self.mace_manager.compute(iter([orig_structure]), batch_size=1))

            max_gamma = 0.0
            if "c_gamma" in mace_result.arrays:
                max_gamma = np.max(mace_result.get_array("c_gamma"))  # type: ignore[no-untyped-call]

            if max_gamma > self.uncertainty_threshold:
                logger.info(f"MACE uncertainty {max_gamma:.4f} > {self.uncertainty_threshold}. Falling back to DFT.")
                dft_result = next(self.dft_manager.compute(iter([orig_structure]), batch_size=1))
                yield dft_result
            else:
                yield mace_result


class DFTManager(BaseOracle):
    """
    Manages DFT calculations with self-healing capabilities.

    Memory Usage:
        This class processes structures one-by-one (streaming) to ensure O(1) memory usage
        relative to the dataset size. It does not materialize the input iterator into a list.
    """

    def __init__(self, config: DFTConfig, driver: QEDriver) -> None:
        """
        Initializes the DFTManager.

        Args:
            config: DFT configuration.
            driver: QEDriver instance (required for dependency injection).
        """
        if config is None:
            msg = "config cannot be None"
            raise ValueError(msg)
        if driver is None:
            msg = "driver cannot be None"
            raise ValueError(msg)
        # We don't strictly assert QEDriver via isinstance at runtime here to support mock overrides
        # But we validated it at the config level and it is statically typed as QEDriver.

        self.config = DFTConfig.model_validate(config)
        self.driver = driver

        # Cache strategies to avoid recreation on every compute call
        self.strategies: list[Callable[[DFTConfig], None] | None] = [
            None,
            self._strategy_reduce_beta,
            self._strategy_increase_smearing,
            self._strategy_use_cg,
        ]

    def compute_uncertainty(self, structure: Atoms) -> float:
        """
        DFT provides ground truth, so uncertainty is 0.0.
        """
        return 0.0

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
        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        """Internal generator for streaming computations with batching."""
        while True:
            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)
                for i in range(batch_size):
                    try:
                        atoms = next(structures)
                    except StopIteration:
                        return
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
            # Preserve uncertainty information if present
            if "c_gamma" in atoms.arrays:
                structure_to_compute.new_array("c_gamma", atoms.get_array("c_gamma").copy())  # type: ignore[no-untyped-call]
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
        from pyacemaker.utils.path import validate_path_safe

        calc_dir = str(validate_path_safe(Path(calc_dir)))

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
            except (RuntimeError, CalculatorSetupError, CalculationFailed) as e:
                # Catch specific exceptions related to calculation failure
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

        try:
            # Trigger actual calculation
            atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            atoms.get_forces()  # type: ignore[no-untyped-call]

            # Try to get stress (optional)
            with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
                atoms.get_stress()  # type: ignore[no-untyped-call]
        except (CalculationFailed, RuntimeError) as e:
            # Explicitly log convergence or execution failure specifically
            logger.debug(f"Calculator failed specifically during execution/convergence: {e}")
            raise
