import threading
from unittest.mock import MagicMock

import pytest

from pyacemaker.core.evaluator import TwoTierEvaluator
from pyacemaker.core.exceptions import MDHaltInterrupt


def test_evaluator_thermal_noise() -> None:
    evaluator = TwoTierEvaluator(threshold_call_dft=0.5, threshold_add_train=0.2, smooth_steps=3)
    lmp_mock = MagicMock()

    # Step 1: exceed threshold -> exceedances = 1
    lmp_mock.extract_variable.return_value = 0.6
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 1
    lmp_mock.command.assert_not_called()

    # Step 2: below threshold (thermal noise) -> reset to 0
    lmp_mock.extract_variable.return_value = 0.4
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 0
    lmp_mock.command.assert_not_called()


def test_evaluator_trigger_halt() -> None:
    evaluator = TwoTierEvaluator(threshold_call_dft=0.5, threshold_add_train=0.2, smooth_steps=3)
    lmp_mock = MagicMock()

    # Mocks
    class MockVarTracker:
        def __init__(self) -> None:
            self.calls = 0

        def get_var(self, arg: str) -> float:
            if arg == "max_g":
                vals = [0.6, 0.7, 0.8, 0.8]
                calls = self.calls
                self.calls += 1
                if calls >= len(vals):
                    return vals[-1]
                return vals[calls]
            if arg == "step":
                return 500.0
            if arg == "atoms":
                return 2.0
            return 0.0

    tracker = MockVarTracker()
    lmp_mock.extract_variable.side_effect = tracker.get_var

    import ctypes
    array_type = ctypes.c_double * 2
    c_array = array_type(0.1, 0.3)
    ptr = ctypes.cast(c_array, ctypes.c_void_p).value
    lmp_mock.extract_compute.return_value = ptr

    # Step 1
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 1

    # Step 2
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 2

    # Step 3 -> trigger
    with pytest.raises(MDHaltInterrupt) as excinfo:
        evaluator.evaluate(lmp_mock)

    assert excinfo.value.step == 500
    assert excinfo.value.epicenter_indices == [2]
    lmp_mock.command.assert_called_with("variable trigger_halt string true")


def test_evaluator_exception_handling() -> None:
    evaluator = TwoTierEvaluator(threshold_call_dft=0.5, threshold_add_train=0.2, smooth_steps=3)
    lmp_mock = MagicMock()
    lmp_mock.extract_variable.side_effect = ValueError("Missing variable")

    with pytest.raises(RuntimeError, match="TwoTierEvaluator encountered an error"):
        evaluator.evaluate(lmp_mock)


def test_evaluator_concurrency() -> None:
    """Cycle 04: Evaluator thread safety."""
    evaluator = TwoTierEvaluator(threshold_call_dft=0.5, threshold_add_train=0.2, smooth_steps=100)

    def worker() -> None:
        lmp_mock = MagicMock()
        lmp_mock.extract_variable.return_value = 0.6
        evaluator.evaluate(lmp_mock)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert evaluator.consecutive_exceedances == 50
