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
    manager = DFTManager(mock_dft_config, driver=fake_driver)  # type: ignore[arg-type]

    # Verify generator behavior with next() instead of list()
    generator = manager.compute(iter([atoms]))
    result = next(generator)

    assert result.get_potential_energy() == TEST_ENERGY_GENERIC  # type: ignore[no-untyped-call]

    # ProcessPoolExecutor copies state, so we can't easily assert on fake_driver call_count
    # Verify generator returned the correctly calculated atoms object instead.
    assert result.get_potential_energy() == TEST_ENERGY_GENERIC


def test_dft_manager_self_healing(
    mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test self-healing mechanism."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Because ProcessPoolExecutor executes in a separate process, mocking stateful objects like `FakeDriver`
    # is difficult. We will mock `ProcessPoolExecutor` itself to test the loop logic.
    from concurrent.futures import Future

    class DummyFuture(Future):  # type: ignore[type-arg]
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:
            # We track the call count at the class level because DummyExecutor is instantiated fresh each loop
            pass

        def __enter__(self) -> "DummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> DummyFuture:
            DummyExecutor.call_count += 1
            if DummyExecutor.call_count == 1:
                return DummyFuture(None, RuntimeError("Setup failed"))

            calc = MockCalculator(fail_count=0)
            atoms = args[1]
            atoms.calc = calc
            atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            return DummyFuture(calc, None)

    DummyExecutor.call_count = 0

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummyExecutor)

    fake_driver = FakeDriver()
    manager = DFTManager(mock_dft_config, driver=fake_driver)  # type: ignore[arg-type]

    gen = manager.compute(iter([atoms]))
    result = next(gen)

    assert result.get_potential_energy() == TEST_ENERGY_GENERIC  # type: ignore[no-untyped-call]


def test_dft_manager_fatal_error(
    mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test fatal error after exhausting retries."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    from concurrent.futures import Future

    class DummyFuture(Future):  # type: ignore[type-arg]
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "DummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> DummyFuture:
            return DummyFuture(None, RuntimeError("Always fails"))

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummyExecutor)

    fake_driver = FakeDriver(calcs=MockCalculator(fail_count=100))

    manager = DFTManager(mock_dft_config, driver=fake_driver)  # type: ignore[arg-type]

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

    class DummyFuture(Future):  # type: ignore[type-arg]
        def __init__(self, result_value: Any, exception: Any = None) -> None:
            super().__init__()
            self._result_value = result_value
            self._exception = exception

        def result(self, timeout: float | None = None) -> Any:
            return self._result_value, self._exception

    class DummyExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> "DummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> DummyFuture:
            return DummyFuture(None, RuntimeError("CalculatorSetupError"))

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummyExecutor)

    fake_driver = FakeDriver(calcs=MockCalculator(setup_error=True))

    manager = DFTManager(mock_dft_config, driver=fake_driver)  # type: ignore[arg-type]

    gen = manager.compute(iter([atoms]))
    with pytest.raises(OracleError, match="Oracle calculation failed"):
        next(gen)

    # Should retry even on setup error if it's considered transient or parameter based?
    # Spec says "JobFailedException" (RuntimeError). Implementation catches (RuntimeError, CalculatorSetupError).
    # So it should retry.


def test_dft_manager_strategies(mock_dft_config: DFTConfig) -> None:
    """Test that strategies are correctly defined."""
    manager = DFTManager(mock_dft_config)
    strategies = manager._get_strategies()

    assert len(strategies) > 0
    assert strategies[0] is None  # First attempt is vanilla

    # Strategy 1: Reduce Beta
    strat_beta = strategies[1]
    assert strat_beta is not None
    config_copy = mock_dft_config.model_copy()
    original_beta = config_copy.mixing_beta
    strat_beta(config_copy)
    assert config_copy.mixing_beta == original_beta * 0.5

    # Strategy 2: Increase Smearing
    strat_smearing = strategies[2]
    assert strat_smearing is not None
    config_copy = mock_dft_config.model_copy()
    original_smearing = config_copy.smearing_width
    strat_smearing(config_copy)
    assert config_copy.smearing_width == original_smearing * 2.0

    # Strategy 3: CG Diagonalization
    strat_cg = strategies[3]
    assert strat_cg is not None
    config_copy = mock_dft_config.model_copy()
    strat_cg(config_copy)
    assert config_copy.diagonalization == "cg"


def test_dft_manager_invalid_input(mock_dft_config: DFTConfig) -> None:
    """Test compute raises TypeError for non-iterator input."""
    manager = DFTManager(mock_dft_config)
    atoms_list = [Atoms("H")]

    # Check that it raises TypeError immediately upon calling compute (before next)
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        manager.compute(atoms_list)  # type: ignore[arg-type]

    # Explicitly check None
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        manager.compute(None)  # type: ignore[arg-type]


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

    manager = DFTManager(mock_dft_config, driver=fake_driver)  # type: ignore[arg-type]

    atoms = Atoms("H", positions=[[0, 0, 0]])
    # Must be iterator
    gen = manager.compute(iter([atoms]))
    result = next(gen)

    # Check if result is the embedded one
    # DFTManager.compute yields the result of _compute_single(embedded_atoms)
    # _compute_single returns the atom object passed to it (which is embedded_atoms)
    assert result == embedded_atoms
