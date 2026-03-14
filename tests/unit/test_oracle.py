from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.oracle import DFTManager, MACEManager, TieredOracle
from pyacemaker.domain_models.dft import DFTConfig
from pyacemaker.domain_models.workflow import ActiveLearningThresholds

TEST_ENERGY_GENERIC = -42.0


class MockCalculator(Calculator):
    implemented_properties: list[str] = ["energy", "forces"]  # noqa: RUF012

    def __init__(self, fail_count: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]
        self.fail_count = fail_count
        self.current_fails = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if self.current_fails < self.fail_count:
            self.current_fails += 1
            msg = "Mock failure"
            raise RuntimeError(msg)

        if atoms is None:
            return
        super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]
        n_atoms = len(atoms)
        self.results["energy"] = TEST_ENERGY_GENERIC * n_atoms
        self.results["forces"] = np.ones((n_atoms, 3)) * 0.1


class FakeDriver:
    def __init__(self, **kwargs: Any) -> None:
        pass


class DummyMaceCalc(Calculator):
    implemented_properties: list[str] = ["energy", "forces"]  # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]
        self.models = [1, 2]  # Dummy to trigger node variance extraction block

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if atoms is None:
            return
        super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]
        n_atoms = len(atoms)
        self.results["energy"] = -10.0 * n_atoms
        self.results["forces"] = np.ones((n_atoms, 3)) * 0.1


def get_safe_test_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    pot_dir = tmp_path / "potentials"
    monkeypatch.setattr("pyacemaker.domain_models.defaults.DEFAULT_POTENTIALS_DIR", str(pot_dir))
    pot_dir.mkdir(parents=True, exist_ok=True)
    return pot_dir / "model.model"


def test_macemanager_initialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = get_safe_test_model_path(monkeypatch, tmp_path)
    model_path.touch()

    with patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))
        assert manager.is_initialized

    model_path.unlink()


def test_macemanager_initialization_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test self-healing mechanism."""

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
            self.call_count = 0

        def __enter__(self) -> "DummyExecutor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> DummyFuture:
            self.call_count += 1
            if self.call_count == 1:
                msg = "Setup failed"
                return DummyFuture(None, RuntimeError(msg))

            calc = MockCalculator(fail_count=0)
            atoms = args[1]
            atoms.calc = calc
            atoms.get_potential_energy()
            return DummyFuture(calc, None)

    executor = DummyExecutor(max_workers=1)
    executor.call_count = 0
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", lambda max_workers: executor)

    model_path = get_safe_test_model_path(monkeypatch, tmp_path)
    model_path.touch()

    with (
        patch(
            "mace.calculators.mace_mp", side_effect=Exception("Simulated Initialization Failure")
        ),
        pytest.raises(OracleError),
    ):
        MACEManager(str(model_path))

    model_path.unlink()


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

    manager = DFTManager(mock_dft_config, driver=FakeDriver())

    gen = manager.compute(iter([atoms]))
    with pytest.raises(OracleError):
        next(gen)


def test_tiered_oracle_initialization() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)
    assert oracle.mace == mock_mace
    assert oracle.dft == mock_dft

    with pytest.raises(ValueError, match="MACEManager must be provided"):
        TieredOracle(mace_manager=None, dft_manager=mock_dft, thresholds=thresholds)

    with pytest.raises(ValueError, match="DFTManager cannot be None"):
        TieredOracle(mace_manager=mock_mace, dft_manager=None, thresholds=thresholds)


def test_tiered_oracle_compute_below_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    atoms_result = Atoms("H")
    atoms_result.new_array("c_gamma", np.array([0.01]))
    mock_mace.compute.return_value = iter([atoms_result])

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)

    atoms_input = Atoms("H")
    result_iter = oracle.compute(iter([atoms_input]))
    result = next(result_iter)

    assert result == atoms_result
    mock_mace.compute.assert_called_once()
    mock_dft.compute.assert_not_called()


def test_tiered_oracle_compute_boundary_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    # Exact boundary edge case
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    atoms_mace_result = Atoms("H")
    atoms_mace_result.new_array("c_gamma", np.array([0.05]))  # type: ignore[no-untyped-call]

    mock_mace.compute.return_value = iter([atoms_mace_result])
    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)

    result = next(oracle.compute(iter([Atoms("H")])))

    # Should NOT fall back to DFT because it is <= threshold (not strictly >)
    assert result == atoms_mace_result
    mock_dft.compute.assert_not_called()


def test_tiered_oracle_compute_above_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    atoms_mace_result = Atoms("H")
    atoms_mace_result.new_array("c_gamma", np.array([0.1]))  # type: ignore[no-untyped-call]

    atoms_dft_result = Atoms("H")

    mock_mace.compute.return_value = iter([atoms_mace_result])
    mock_dft.compute.return_value = iter([atoms_dft_result])

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)

    atoms_input = Atoms("H")
    result_iter = oracle.compute(iter([atoms_input]))
    result = next(result_iter)

    assert result == atoms_dft_result
    assert result.has("c_gamma")
    assert np.array_equal(result.get_array("c_gamma"), np.array([0.1]))

    mock_mace.compute.assert_called_once()
    mock_dft.compute.assert_called_once()


def test_tiered_oracle_compute_invalid_input() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        oracle.compute([Atoms("H")])
