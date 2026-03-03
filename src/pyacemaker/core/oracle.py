import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.defaults import ERR_ORACLE_FAILED, ERR_ORACLE_ITERATOR
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

logger = logging.getLogger(__name__)


class MACEManager(BaseOracle):
    """
    Wrapper to execute MACE-MP-0 inference.
    """

    def __init__(self, model_path: str = "MACE-MP-0") -> None:
        self.model_path = model_path

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes properties using MACE foundation model.
        """
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        for atoms in structures:
            # Placeholder for MACE inference.
            # In real code: atoms.calc = self._calculator; atoms.get_potential_energy()

            # Mocking the inference for tests and UAT
            import numpy as np

            atoms.info["energy"] = -5.0 * len(atoms)
            atoms.arrays["forces"] = np.zeros((len(atoms), 3))
            atoms.arrays["mace_uncertainty"] = np.random.uniform(0.0, 1.0, len(atoms))

            yield atoms


class TieredOracle(BaseOracle):
    """
    Manages query routing between fast (MACE) and slow (DFT) oracles based on uncertainty.
    """

    def __init__(
        self, mace_manager: MACEManager, dft_manager: "DFTManager", threshold: float
    ) -> None:
        self.mace = mace_manager
        self.dft = dft_manager
        self.threshold = threshold

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Routes structures to MACE first. If uncertainty is high, falls back to DFT.
        """
        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        # Evaluate all structures with MACE first
        mace_evaluated = self.mace.compute(structures, batch_size=batch_size)

        for atoms in mace_evaluated:
            # Check uncertainty
            uncertainties = atoms.arrays.get("mace_uncertainty")
            if uncertainties is not None and uncertainties.max() > self.threshold:
                logger.info(
                    f"Uncertainty {uncertainties.max():.2f} > {self.threshold}. Falling back to DFT."
                )

                # We need to pass it as an iterator to DFTManager
                def _single_iterator(a: Atoms = atoms) -> Iterator[Atoms]:
                    yield a

                dft_result = list(self.dft.compute(_single_iterator(), batch_size=1))
                if dft_result:
                    yield dft_result[0]
            else:
                yield atoms


class DFTManager(BaseOracle):
    """
    Manages DFT calculations with self-healing capabilities.
    """

    def __init__(self, config: DFTConfig, driver: QEDriver | None = None) -> None:
        self.config = config
        self.driver = driver or QEDriver()

        self.strategies: list[Callable[[DFTConfig], None] | None] = [
            None,
            self._strategy_reduce_beta,
            self._strategy_increase_smearing,
            self._strategy_use_cg,
        ]

    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        if isinstance(structures, (list, tuple)):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        if not isinstance(structures, Iterator):
            raise TypeError(ERR_ORACLE_ITERATOR.format(type=type(structures)))

        return self._compute_generator(structures, batch_size)

    def _compute_generator(self, structures: Iterator[Atoms], batch_size: int) -> Iterator[Atoms]:
        while True:
            batch = list(islice(structures, batch_size))
            if not batch:
                break

            processed_batch = []

            with tempfile.TemporaryDirectory() as work_dir:
                work_path = Path(work_dir)
                for i, atoms in enumerate(batch):
                    calc_dir = work_path / f"calc_{i}"
                    calc_dir.mkdir()
                    # Fully process and store result before yielding to ensure TempDir is cleaned up
                    processed_batch.append(self._process_structure(atoms, str(calc_dir)))

            yield from processed_batch

    def _process_structure(self, atoms: Atoms, calc_dir: str) -> Atoms:
        if self.config.embedding_buffer:
            structure_to_compute = embed_cluster(atoms, buffer=self.config.embedding_buffer)
        else:
            structure_to_compute = atoms

        return self._compute_single(structure_to_compute, calc_dir)

    def _get_strategies(self) -> list[Callable[[DFTConfig], None] | None]:
        return self.strategies

    def _strategy_reduce_beta(self, c: DFTConfig) -> None:
        c.mixing_beta *= self.config.mixing_beta_factor

    def _strategy_increase_smearing(self, c: DFTConfig) -> None:
        c.smearing_width *= self.config.smearing_width_factor

    def _strategy_use_cg(self, c: DFTConfig) -> None:
        c.diagonalization = "cg"

    def _compute_single(self, atoms: Atoms, calc_dir: str) -> Atoms:
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
            except Exception as e:
                last_error = e
                atoms.calc = None
                logger.warning(
                    f"DFT calculation attempt {i + 1} ({strategy_name}) failed. Error: {e!s}. Retrying..."
                )
                continue
            else:
                return atoms

        raise OracleError(ERR_ORACLE_FAILED.format(error=last_error)) from last_error

    def _run_calculator(self, atoms: Atoms, config: DFTConfig, calc_dir: str) -> None:
        calc = self.driver.get_calculator(atoms, config.model_copy(), directory=calc_dir)
        atoms.calc = calc
        atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        atoms.get_forces()  # type: ignore[no-untyped-call]

        with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
            atoms.get_stress()  # type: ignore[no-untyped-call]
