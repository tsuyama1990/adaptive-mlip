from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ase import Atoms

from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.conftest import MockCalculator, create_dummy_pseudopotentials
from tests.constants import TEST_ENERGY_GENERIC


@pytest.fixture
def mock_dft_config(dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["H"])

    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF"},
    )


class FakeDriver:
    """Fake driver to be picklable for ProcessPoolExecutor"""

    def __init__(self, calcs: list[MockCalculator] | MockCalculator | None = None) -> None:
        self.calcs = (
            calcs
            if isinstance(calcs, list)
            else [calcs]
            if calcs
            else [MockCalculator(fail_count=0)]
        )
        self.call_count = 0
        self.call_args_list: list[tuple[Any, Any]] = []

    def get_calculator(self, atoms: Atoms, config: Any, directory: str) -> Any:
        self.call_args_list.append(((atoms, config), {"directory": directory}))
        calc = self.calcs[self.call_count] if self.call_count < len(self.calcs) else self.calcs[-1]
        self.call_count += 1
        return calc


def test_dft_manager_compute_success(mock_dft_config: DFTConfig) -> None:
    """Test successful computation using dependency injection."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Create Fake Driver
    fake_driver = FakeDriver()

    # Inject fake driver
    manager = DFTManager(mock_dft_config, driver=fake_driver)

    # Verify generator behavior with next() instead of list()
    generator = manager.compute(iter([atoms]))
    result = next(generator)

    assert result.get_potential_energy() == TEST_ENERGY_GENERIC

    # ProcessPoolExecutor copies state, so we can't easily assert on fake_driver call_count
    # Verify generator returned the correctly calculated atoms object instead.
    assert result.get_potential_energy() == TEST_ENERGY_GENERIC


def test_dft_manager_self_healing(
    mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test self-healing mechanism by simply mocking the calculator runner function."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    call_count = 0

    def mock_run_calculator_process(
        driver: Any, atoms_to_calc: Atoms, config: DFTConfig, calc_dir: str
    ) -> tuple[Any, Exception | None]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, RuntimeError("First attempt failed")
        # Second attempt succeeds
        calc = MockCalculator(fail_count=0)
        atoms_to_calc.calc = calc
        atoms_to_calc.get_potential_energy()  # type: ignore[no-untyped-call]
        return calc, None

    monkeypatch.setattr(
        "pyacemaker.core.oracle._run_calculator_process", mock_run_calculator_process
    )

    # Inject a synchronous executor to avoid process boundaries making tests flaky
    from concurrent.futures import Future

    class SyncDummyFuture(Future[Any]):
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class SyncDummyExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "SyncDummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            res, exc = fn(*args, **kwargs)
            return SyncDummyFuture(res, exc)

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", SyncDummyExecutor)

    fake_driver = FakeDriver()
    manager = DFTManager(mock_dft_config, driver=fake_driver)

    gen = manager.compute(iter([atoms]))
    result = next(gen)

    assert result.get_potential_energy() == TEST_ENERGY_GENERIC


def test_dft_manager_fatal_error(
    mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test fatal error after exhausting retries."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    from concurrent.futures import Future

    def mock_run_calculator_process(
        driver: Any, atoms_to_calc: Atoms, config: DFTConfig, calc_dir: str
    ) -> tuple[Any, Exception | None]:
        return None, RuntimeError("Always fails")

    monkeypatch.setattr(
        "pyacemaker.core.oracle._run_calculator_process", mock_run_calculator_process
    )

    class SyncDummyFuture(Future[Any]):
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class SyncDummyExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "SyncDummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            res, exc = fn(*args, **kwargs)
            return SyncDummyFuture(res, exc)

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", SyncDummyExecutor)

    fake_driver = FakeDriver(calcs=MockCalculator(fail_count=100))

    manager = DFTManager(mock_dft_config, driver=fake_driver)

    # Now raises OracleError
    # Use next() to trigger execution
    gen = manager.compute(iter([atoms]))
    with pytest.raises(OracleError, match="Oracle calculation failed"):
        next(gen)

    # In tests, if ProcessPoolExecutor is used, state updates inside _run_calculator
    # (like mock call counts on self.driver) are not reflected back in the main process
    # because they happen in a separate process space. This is a limitation of testing
    # ProcessPoolExecutor.
    # We will just assert that the code raises the correct exception.


def test_dft_manager_setup_error(
    mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test handling of CalculatorSetupError."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    from concurrent.futures import Future

    def mock_run_calculator_process(
        driver: Any, atoms_to_calc: Atoms, config: DFTConfig, calc_dir: str
    ) -> tuple[Any, Exception | None]:
        return None, RuntimeError("CalculatorSetupError")

    monkeypatch.setattr(
        "pyacemaker.core.oracle._run_calculator_process", mock_run_calculator_process
    )

    class SyncDummyFuture(Future[Any]):
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class SyncDummyExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "SyncDummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            res, exc = fn(*args, **kwargs)
            return SyncDummyFuture(res, exc)

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", SyncDummyExecutor)

    fake_driver = FakeDriver(calcs=MockCalculator(setup_error=True))

    manager = DFTManager(mock_dft_config, driver=fake_driver)

    gen = manager.compute(iter([atoms]))
    with pytest.raises(OracleError, match="Oracle calculation failed"):
        next(gen)

    # Should retry even on setup error if it's considered transient or parameter based?
    # Spec says "JobFailedException" (RuntimeError). Implementation catches (RuntimeError, CalculatorSetupError).
    # So it should retry.


def test_dft_manager_strategies(
    mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that strategies are applied correctly via the compute public API."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    applied_configs = []

    # Mock the internal calculator runner to just capture the config it was given
    def mock_run_calculator_process(
        driver: Any, atoms_to_calc: Atoms, config: DFTConfig, calc_dir: str
    ) -> tuple[Any, Exception | None]:
        # Append a deepcopy to capture the state at this exact point in time
        applied_configs.append(config.model_copy())
        # Always fail to trigger all strategies
        return None, RuntimeError("Intentional failure to trigger strategies")

    monkeypatch.setattr(
        "pyacemaker.core.oracle._run_calculator_process", mock_run_calculator_process
    )

    # Use a dummy executor that just runs the function synchronously for tests
    from concurrent.futures import Future

    class SyncDummyFuture(Future[Any]):
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class SyncDummyExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "SyncDummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            res, exc = fn(*args, **kwargs)
            return SyncDummyFuture(res, exc)

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", SyncDummyExecutor)

    manager = DFTManager(mock_dft_config)

    with pytest.raises(OracleError):
        list(manager.compute(iter([atoms])))

    # 4 attempts total: Vanilla, Reduce Beta, Increase Smearing, Use CG
    assert len(applied_configs) == 4

    # Verify configs were correctly mutated by strategies incrementally
    assert applied_configs[0].mixing_beta == mock_dft_config.mixing_beta
    assert applied_configs[0].smearing_width == mock_dft_config.smearing_width
    assert applied_configs[0].diagonalization == mock_dft_config.diagonalization

    assert (
        applied_configs[1].mixing_beta
        == mock_dft_config.mixing_beta * mock_dft_config.mixing_beta_factor
    )
    assert applied_configs[1].smearing_width == mock_dft_config.smearing_width
    assert applied_configs[1].diagonalization == mock_dft_config.diagonalization

    assert (
        applied_configs[2].mixing_beta
        == mock_dft_config.mixing_beta * mock_dft_config.mixing_beta_factor
    )
    assert (
        applied_configs[2].smearing_width
        == mock_dft_config.smearing_width * mock_dft_config.smearing_width_factor
    )
    assert applied_configs[2].diagonalization == mock_dft_config.diagonalization

    assert (
        applied_configs[3].mixing_beta
        == mock_dft_config.mixing_beta * mock_dft_config.mixing_beta_factor
    )
    assert (
        applied_configs[3].smearing_width
        == mock_dft_config.smearing_width * mock_dft_config.smearing_width_factor
    )
    assert applied_configs[3].diagonalization == "cg"


def test_dft_manager_invalid_input(mock_dft_config: DFTConfig) -> None:
    """Test compute raises TypeError for non-iterator input."""
    manager = DFTManager(mock_dft_config)
    atoms_list = [Atoms("H")]

    # Check that it raises TypeError immediately upon calling compute (before next)
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        manager.compute(atoms_list)

    # Explicitly check None
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        manager.compute(None)


def test_dft_manager_empty_iterator(mock_dft_config: DFTConfig) -> None:
    """Test compute handles empty iterator correctly safely."""
    manager = DFTManager(mock_dft_config)
    empty_iter: Iterator[Atoms] = iter([])

    # Explicit loop without list() materialization for safety
    # Use deque(..., maxlen=0) to consume iterator efficiently
    from collections import deque

    deque(manager.compute(empty_iter), maxlen=0)


def test_dft_manager_embedding(mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that embedding is applied when configured."""
    from pyacemaker.core.oracle import DFTManager

    # Configure embedding buffer
    mock_dft_config.embedding_buffer = 5.0

    # Mock embed_cluster
    # It must be picklable too, or not used here.
    # We will just patch `embed_cluster` with a simple function
    embedded_atoms = Atoms("H", cell=[20, 20, 20], pbc=True)

    def fake_embed(*args: Any, **kwargs: Any) -> Atoms:
        return embedded_atoms

    monkeypatch.setattr("pyacemaker.core.oracle.embed_cluster", fake_embed)

    # Mock Driver
    fake_driver = FakeDriver(calcs=MockCalculator(fail_count=0))

    manager = DFTManager(mock_dft_config, driver=fake_driver)

    atoms = Atoms("H", positions=[[0, 0, 0]])
    # Must be iterator
    gen = manager.compute(iter([atoms]))
    result = next(gen)

    # Check if result is the embedded one
    # DFTManager.compute yields the result of _compute_single(embedded_atoms)
    # _compute_single returns the atom object passed to it (which is embedded_atoms)
    assert result == embedded_atoms
