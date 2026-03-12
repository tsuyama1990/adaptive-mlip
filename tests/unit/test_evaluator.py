from unittest.mock import MagicMock

import pytest

from pyacemaker.core.evaluator import TwoTierEvaluator


def test_evaluator_thermal_noise():
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


def test_evaluator_trigger_halt():
    evaluator = TwoTierEvaluator(threshold_call_dft=0.5, threshold_add_train=0.2, smooth_steps=3)
    lmp_mock = MagicMock()

    # Step 1
    lmp_mock.extract_variable.return_value = 0.6
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 1

    # Step 2
    lmp_mock.extract_variable.return_value = 0.7
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 2

    # Step 3 -> trigger
    lmp_mock.extract_variable.return_value = 0.8
    evaluator.evaluate(lmp_mock)
    assert evaluator.consecutive_exceedances == 3
    lmp_mock.command.assert_called_with("variable trigger_halt string true")


def test_evaluator_exception_handling():
    evaluator = TwoTierEvaluator(threshold_call_dft=0.5, threshold_add_train=0.2, smooth_steps=3)
    lmp_mock = MagicMock()
    lmp_mock.extract_variable.side_effect = ValueError("Missing variable")

    with pytest.raises(ValueError, match="Missing variable"):
        evaluator.evaluate(lmp_mock)
