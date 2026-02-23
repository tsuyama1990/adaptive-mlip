import contextlib
from collections.abc import Callable, Iterator

from ase import Atoms
from ase.calculators.calculator import CalculatorSetupError, PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.misc import batched


class DFTManager(BaseOracle):
    """
    Manages DFT calculations with self-healing capabilities.
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
        """
        # Process in batches to allow for future parallelization
        for batch in batched(structures, batch_size):
            # Currently serial execution, but grouped by batch
            for atoms in batch:
                yield self._compute_single(atoms)

    def _get_strategies(self) -> list[Callable[[DFTConfig], None] | None]:
        """
        Returns a list of self-healing strategies.
        Each strategy is a callable that modifies the configuration in-place,
        or None (representing the initial attempt with unmodified config).
        """
        # Use multipliers from config
        beta_factor = self.config.mixing_beta_factor
        smearing_factor = self.config.smearing_width_factor

        return [
            None,  # First attempt: no change
            lambda c: setattr(c, "mixing_beta", c.mixing_beta * beta_factor),
            lambda c: setattr(c, "smearing_width", c.smearing_width * smearing_factor),
            lambda c: setattr(c, "diagonalization", "cg"),
        ]

    def _compute_single(self, atoms: Atoms) -> Atoms:
        """
        Runs calculation for a single structure with retries.
        """
        # Create a mutable copy of config
        current_config = self.config.model_copy()

        strategies = self._get_strategies()
        last_error: Exception | None = None

        for _, strategy in enumerate(strategies):
            if strategy:
                strategy(current_config)

            try:
                # Create calculator with current config
                # Pass a copy to avoid mutation issues if driver modifies it or for test stability
                calc = self.driver.get_calculator(atoms, current_config.model_copy())
                atoms.calc = calc

                # Trigger calculation
                atoms.get_potential_energy()
                atoms.get_forces()

                # Try to get stress, ignore if not implemented
                with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
                    atoms.get_stress()

            except (RuntimeError, CalculatorSetupError) as e:
                last_error = e
                continue
            else:
                return atoms

        # If we reach here, all attempts failed
        msg = f"DFT calculation failed after {len(strategies)} attempts. Last error: {last_error}"
        raise OracleError(msg) from last_error
