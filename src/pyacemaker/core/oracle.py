import contextlib
from collections.abc import Callable, Iterator

from ase import Atoms
from ase.calculators.calculator import CalculatorSetupError, PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.interfaces.qe_driver import QEDriver


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

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes DFT properties for stream of structures.

        Args:
            structures: Iterator of Atoms objects.
            batch_size: Ignored in this implementation to ensure strict streaming (one-by-one).
                        Kept for interface compatibility.

        Yields:
            Atoms objects with computed properties.

        Raises:
            OracleError: If a calculation fails fatally.
            TypeError: If structures is not an iterator (to prevent memory leaks from huge lists).
        """
        # Validate that structures is an iterator to enforce O(1) memory usage contract
        # isinstance check against Iterator (from collections.abc) might be tricky with some generators?
        # But iter(list) returns 'list_iterator' which inherits from Iterator.
        # However, a list is Iterable but NOT Iterator.
        # Let's ensure we import Iterator from collections.abc correctly.
        if not isinstance(structures, Iterator):
            msg = f"Input 'structures' must be an Iterator (got {type(structures)}). Use iter() to create one."
            raise TypeError(msg)

        # Strict streaming: Process one by one.
        # We do NOT use batched() here to avoid even small batch materialization in memory
        # as per strict audit requirements.
        iterator_empty = True

        for atoms in structures:
            iterator_empty = False
            yield self._compute_single(atoms)

        if iterator_empty:
             # Audit requirement: "Add explicit handling for empty iterators with appropriate error messages."
             # Returning empty iterator is valid, but logging helps debug.
             import warnings
             warnings.warn("Oracle received empty iterator. No calculations performed.", UserWarning, stacklevel=2)

    def _get_strategies(self) -> list[Callable[[DFTConfig], None] | None]:
        """
        Returns a list of self-healing strategies.
        """
        # Use multipliers from config
        beta_factor = self.config.mixing_beta_factor
        smearing_factor = self.config.smearing_width_factor

        # Define strategies as named functions for clarity/maintainability
        def reduce_beta(c: DFTConfig) -> None:
            c.mixing_beta *= beta_factor

        def increase_smearing(c: DFTConfig) -> None:
            c.smearing_width *= smearing_factor

        def use_cg(c: DFTConfig) -> None:
            c.diagonalization = "cg"

        return [
            None,  # First attempt: no change
            reduce_beta,
            increase_smearing,
            use_cg,
        ]

    def _compute_single(self, atoms: Atoms) -> Atoms:
        """
        Runs calculation for a single structure with retries and self-healing strategies.

        Args:
            atoms: The atomic structure to calculate.

        Returns:
            Atoms object with calculated properties attached.

        Raises:
            OracleError: If calculation fails after all retries and strategies.
        """
        current_config = self.config.model_copy()
        strategies = self._get_strategies()
        last_error: Exception | None = None

        for strategy in strategies:
            if strategy:
                strategy(current_config)

            try:
                self._run_calculator(atoms, current_config)
            except (RuntimeError, CalculatorSetupError) as e:
                last_error = e
                atoms.calc = None  # Clean up failed calculator
                continue
            else:
                return atoms

        msg = f"DFT calculation failed after {len(strategies)} attempts."
        raise OracleError(msg) from last_error

    def _run_calculator(self, atoms: Atoms, config: DFTConfig) -> None:
        """Helper to run a single calculation attempt."""
        # Create new calculator for clean state
        calc = self.driver.get_calculator(atoms, config.model_copy())
        atoms.calc = calc

        # Trigger actual calculation
        atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        atoms.get_forces()  # type: ignore[no-untyped-call]

        # Try to get stress (optional)
        with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
            atoms.get_stress()  # type: ignore[no-untyped-call]
