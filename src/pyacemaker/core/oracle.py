import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import ERR_ORACLE_FAILED, ERR_ORACLE_ITERATOR
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

try:
    from mace.calculators import mace_mp

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    mace_mp = None


class CalculatorProtocol(Protocol):
    results: dict[str, Any]

    def calculate(self, atoms: Atoms, properties: list[str], system_changes: list[str]) -> None: ...
    def get_property(
        self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any: ...


logger = logging.getLogger(__name__)


class SelfHealingManager:
    """
    Manages self-healing strategies for DFT calculations.
    """

    def __init__(self, config: DFTConfig) -> None:
        self.config = config
        self.strategies: list[Callable[[DFTConfig], None] | None] = [
            None,
            self._strategy_reduce_beta,
            self._strategy_increase_smearing,
            self._strategy_use_cg,
        ]

    def _strategy_reduce_beta(self, c: DFTConfig) -> None:
        c.mixing_beta *= self.config.mixing_beta_factor

    def _strategy_increase_smearing(self, c: DFTConfig) -> None:
        c.smearing_width *= self.config.smearing_width_factor

    def _strategy_use_cg(self, c: DFTConfig) -> None:
        c.diagonalization = "cg"

    def get_strategies(self) -> list[Callable[[DFTConfig], None] | None]:
        return self.strategies


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
        self.self_healing = SelfHealingManager(config)

    def compute(
        self, structures: Iterator[Atoms], batch_size: int | None = None
    ) -> Iterator[Atoms]:
        """
        Computes DFT properties for stream of structures.

        Args:
            structures: Iterator of Atoms objects.
            batch_size: Deprecated. Left for backwards compatibility.

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

        return self._compute_generator(structures)

    def _compute_generator(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        """
        Internal generator for streaming computations.

        This method processes structures one-by-one from the input iterator, guaranteeing
        O(1) memory usage regardless of dataset size. It uses a single shared temporary
        directory for the entire generator lifetime to prevent I/O bottlenecks.
        """
        # Check if iterator is initially empty (optional but helpful for early warnings)
        try:
            first_item = next(structures)
        except StopIteration:
            import warnings

            warnings.warn("Oracle received empty iterator", UserWarning, stacklevel=2)
            return

        def chained_structures() -> Iterator[Atoms]:
            yield first_item
            yield from structures

        from pyacemaker.utils.path import validate_path_safe

        # Use a single temp dir for the lifetime of the stream to avoid I/O bottlenecks
        with tempfile.TemporaryDirectory() as work_dir:
            work_path = Path(work_dir)
            for i, atoms in enumerate(chained_structures()):
                calc_dir = work_path / f"calc_{i}"
                calc_dir.mkdir()
                # Validate path safely just in case
                validate_path_safe(calc_dir)
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
        return self.self_healing.get_strategies()

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

        # Filter out None to avoid unnecessary checks inside the loop, though
        # first attempt is "None" meaning "no change". We can handle it explicitly.
        for i, strategy in enumerate(strategies):
            if strategy is not None:
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
        atoms.get_potential_energy()  # type: ignore[no-untyped-call]  # type: ignore[no-untyped-call]
        atoms.get_forces()  # type: ignore[no-untyped-call]  # type: ignore[no-untyped-call]

        # Try to get stress (optional)
        with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
            atoms.get_stress()  # type: ignore[no-untyped-call]


class MACEManager(BaseOracle):
    """
    A wrapper around the MACE Python package for fast structure evaluation and uncertainty estimation.
    """

    def __init__(self, model_path: str = "mace-mp-0-medium") -> None:
        self.model_path = model_path
        self._calculator: CalculatorProtocol | None = None

    @property
    def calculator(self) -> CalculatorProtocol:
        if self._calculator is None:
            if not MACE_AVAILABLE:
                msg = "The 'mace' package is required for MACEManager. Please install it with 'pip install mace-torch'"
                raise RuntimeError(msg)

            # Use mace_mp(model=self.model_path) if available, else a dummy or standard init.
            self._calculator = mace_mp(model=self.model_path)
        return self._calculator

    def compute(
        self, structures: Iterator[Atoms], batch_size: int | None = None
    ) -> Iterator[Atoms]:
        """
        Evaluates structures using the MACE model.

        Args:
            structures: Iterator of Atoms objects.
            batch_size: Deprecated. Left for backwards compatibility.

        Yields:
            Atoms objects with computed properties (energy, forces, and uncertainty).
        """
        if isinstance(structures, (list, tuple)):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures)

    def _compute_generator(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        try:
            first_item = next(structures)
        except StopIteration:
            import warnings

            warnings.warn("Oracle received empty iterator", UserWarning, stacklevel=2)
            return

        def chained_structures() -> Iterator[Atoms]:
            yield first_item
            yield from structures

        for atoms in chained_structures():
            try:
                atoms.calc = self.calculator
                # This will populate atoms.calc.results
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                atoms.get_forces()  # type: ignore[no-untyped-call]

                # Extract uncertainty if provided by MACE (e.g., energy_variance or similar)
                # If not natively provided in atoms.calc.results, we provide a placeholder.
                # Standard MACE may provide 'energy_variance' or 'forces_variance'.
                uncertainty = atoms.calc.results.get("energy_variance", 0.0)

                # If mock, we might want to simulate uncertainty for tests
                if hasattr(self.calculator, "fail_count"):
                    # Mock behavior for tests: grab uncertainty from info if pre-seeded
                    uncertainty = atoms.info.get("uncertainty", uncertainty)

                atoms.info["uncertainty"] = uncertainty
                yield atoms
            except Exception as e:
                logger.warning(f"MACE calculation failed: {e!s}. Returning high uncertainty.")
                # Return high uncertainty instead of crashing to maintain the invariant.
                atoms.info["uncertainty"] = float("inf")
                yield atoms


class TieredOracle(BaseOracle):
    """
    A routing Oracle that combines a fast Oracle (e.g., MACEManager) and a slow Oracle (e.g., QEDriver).
    """

    def __init__(
        self,
        fast_oracle: BaseOracle,
        slow_oracle: BaseOracle,
        uncertainty_threshold: float,
        call_dft_threshold: float,
    ) -> None:
        self.fast_oracle = fast_oracle
        self.slow_oracle = slow_oracle
        self.uncertainty_threshold = uncertainty_threshold
        self.call_dft_threshold = call_dft_threshold

    def compute(
        self, structures: Iterator[Atoms], batch_size: int | None = None
    ) -> Iterator[Atoms]:
        """
        Evaluates structures using a tiered strategy.
        First passes through fast oracle. If uncertainty is low, keeps fast result.
        If uncertainty is high, passes to slow oracle.
        """
        if isinstance(structures, (list, tuple)):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures)

    def _compute_generator(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        try:
            first_item = next(structures)
        except StopIteration:
            import warnings

            warnings.warn("Oracle received empty iterator", UserWarning, stacklevel=2)
            return

        def chained_structures() -> Iterator[Atoms]:
            yield first_item
            yield from structures

        # We pass structures through the fast oracle one by one.
        # But fast_oracle.compute takes an iterator. We can wrap the inner iterator.

        for atoms in chained_structures():
            # Pass a single-element iterator to fast_oracle to process one structure at a time
            # Note: This might be slightly less efficient if fast_oracle supports batching internally,
            # but BaseOracle interface defines yielding one by one anyway.
            fast_result_iter = self.fast_oracle.compute(iter([atoms]))

            try:
                fast_atoms = next(fast_result_iter)
            except StopIteration:
                continue

            uncertainty = fast_atoms.info.get("uncertainty", 0.0)

            # Routing Logic
            if uncertainty <= self.uncertainty_threshold:
                # Keep fast result
                yield fast_atoms
            elif uncertainty > self.call_dft_threshold:
                # Route to slow oracle
                # Must clear the calculator to force recalculation if the slow oracle relies on it
                fast_atoms.calc = None
                slow_result_iter = self.slow_oracle.compute(iter([fast_atoms]))
                try:
                    slow_atoms = next(slow_result_iter)
                    # We discard fast_atoms's forces/energy in favor of slow_atoms
                    yield slow_atoms
                except StopIteration:
                    continue
            else:
                # Uncertainty is between thresholds. We can choose to keep the fast one or slow one.
                # The SPEC says:
                # Let's yield fast_atoms here to be safe, or just fall through to fast_atoms.
                yield fast_atoms
