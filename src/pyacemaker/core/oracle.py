import contextlib
from collections.abc import Iterator

from ase import Atoms
from ase.calculators.calculator import CalculatorSetupError, PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models import DFTConfig
from pyacemaker.interfaces.qe_driver import QEDriver


class DFTManager(BaseOracle):
    """
    Manages DFT calculations with self-healing capabilities.
    """

    def __init__(self, config: DFTConfig) -> None:
        self.config = config
        self.driver = QEDriver()

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes DFT properties for stream of structures.
        """
        # Note: batch_size is ignored for now as we process one by one
        for atoms in structures:
            yield self._compute_single(atoms)

    def _compute_single(self, atoms: Atoms) -> Atoms:
        """
        Runs calculation for a single structure with retries.
        """
        # Create a mutable copy of config
        current_config = self.config.model_copy()

        # Strategy definitions
        # 1. Original (None)
        # 2. Reduced beta
        # 3. Increased smearing
        # 4. Change diagonalization
        strategies = [
            None,
            lambda c: setattr(c, "mixing_beta", c.mixing_beta * 0.5),
            lambda c: setattr(c, "smearing_width", c.smearing_width * 2.0),
            lambda c: setattr(c, "diagonalization", "cg"),
        ]

        last_error: Exception | None = None

        # We need a fresh atoms object if we want to be safe, but modifying atoms.calc is standard.
        # However, if calculation fails, atoms.calc might be in bad state.
        # But get_calculator creates a NEW calculator instance each time.
        # So we just assign atoms.calc = new_calc.

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

                return atoms

            except (RuntimeError, CalculatorSetupError) as e:
                last_error = e
                continue

        # If we reach here, all attempts failed
        msg = f"DFT calculation failed after {len(strategies)} attempts. Last error: {last_error}"
        raise RuntimeError(msg) from last_error
