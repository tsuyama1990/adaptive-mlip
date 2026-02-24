import contextlib
from collections.abc import Callable, Iterator

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster


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
        if isinstance(structures, (list, tuple)):
            msg = "Input 'structures' must be an Iterator, not a list/tuple. Use iter() to avoid memory issues."
            raise TypeError(msg)

        if not isinstance(structures, Iterator):
            msg = f"Input 'structures' must be an Iterator (got {type(structures)}). Use iter() to create one."
            raise TypeError(msg)

        # Strict streaming: Process one by one.
        # We do NOT use batched() here to avoid even small batch materialization in memory
        # as per strict audit requirements.

        # We process items as they come. We track if we processed any to warn if empty.
        # This avoids preemptively consuming the iterator with next(), which can be risky for some streams.

        count = 0
        for atoms in structures:
            count += 1
            yield self._process_structure(atoms)

        if count == 0:
             import warnings
             warnings.warn("Oracle received empty iterator. No calculations performed.", UserWarning, stacklevel=2)

    def _process_structure(self, atoms: Atoms) -> Atoms:
        """
        Applies embedding and computes properties for a single structure.

        Args:
            atoms: The input atomic structure.

        Returns:
            Atoms: The structure with computed properties (energy, forces, stress).
                   If embedding is configured, properties are computed for the embedded cluster.
        """
        # Apply Periodic Embedding if configured
        if self.config.embedding_buffer:
            structure_to_compute = embed_cluster(atoms, buffer=self.config.embedding_buffer)
        else:
            structure_to_compute = atoms

        # Safety check for OOM prevention
        if len(structure_to_compute) > 2000:
            msg = (
                f"Structure too large for DFT ({len(structure_to_compute)} atoms). "
                "Skipping to prevent OOM/Hang."
            )
            raise OracleError(msg)

        return self._compute_single(structure_to_compute)

    def _get_strategies(self) -> list[Callable[[DFTConfig], None] | None]:
        """
        Returns a list of self-healing strategies.

        These strategies are applied sequentially if the initial calculation fails.
        Each strategy modifies the configuration in place to attempt recovery (e.g., reducing mixing beta).

        Returns:
            List of callable strategies (or None for the initial attempt).
        """
        return self.strategies

    def _strategy_reduce_beta(self, c: DFTConfig) -> None:
        c.mixing_beta *= self.config.mixing_beta_factor

    def _strategy_increase_smearing(self, c: DFTConfig) -> None:
        c.smearing_width *= self.config.smearing_width_factor

    def _strategy_use_cg(self, c: DFTConfig) -> None:
        c.diagonalization = "cg"

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
            except Exception as e:
                # Catch all exceptions (RuntimeError, CalculatorSetupError, JobFailedException etc)
                # to ensure self-healing strategies are attempted.
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
