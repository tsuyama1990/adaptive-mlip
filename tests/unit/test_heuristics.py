import pytest

from pyacemaker.domain_models.heuristics import get_heuristics_for_slider


@pytest.mark.parametrize("slider", range(1, 11))
def test_get_heuristics_valid_bounds(slider: int) -> None:
    heuristics = get_heuristics_for_slider(slider, ["Al"])

    # DFT threshold should be between 0.001 and 0.5
    assert 0.001 <= heuristics["md"]["uncertainty_threshold"] <= 0.5

    # Check interval should be between 1 and 100
    assert 1 <= heuristics["md"]["check_interval"] <= 100

    # MD timestep should be between 0.0001 ps and 0.003 ps (0.1 to 3 fs)
    assert 0.0001 <= heuristics["md"]["timestep"] <= 0.003

    # Encut should be clamped
    assert 20.0 <= heuristics["dft"]["encut"] <= 150.0

    # Learning rate clamped
    assert 1e-6 <= heuristics["training"]["pacemaker"]["learning_rate"] <= 0.1

    assert heuristics["dft"]["smearing_type"] == "gaussian"
    assert heuristics["dft"]["smearing_width"] == 0.1


def test_heuristics_monotonically_decreasing_accuracy() -> None:
    """Verify that as the slider goes up (speed -> accuracy), the thresholds drop monotonically."""
    prev_threshold = 100.0
    prev_timestep = 100.0
    prev_interval = 1000

    for slider in range(1, 11):
        h = get_heuristics_for_slider(slider, ["Al"])

        assert h["md"]["uncertainty_threshold"] < prev_threshold
        assert h["md"]["timestep"] < prev_timestep
        assert h["md"]["check_interval"] <= prev_interval

        prev_threshold = h["md"]["uncertainty_threshold"]
        prev_timestep = h["md"]["timestep"]
        prev_interval = h["md"]["check_interval"]


def test_heuristics_contextual_fallback() -> None:
    h = get_heuristics_for_slider(5, ["Pt"])
    assert h["dft"]["smearing_type"] == "mv"
    assert h["dft"]["smearing_width"] == 0.02

    h = get_heuristics_for_slider(5, ["Al", "O"])
    assert h["dft"]["smearing_type"] == "gaussian"
    assert h["dft"]["smearing_width"] == 0.1


def test_heuristics_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="Slider value must be between 1 and 10"):
        get_heuristics_for_slider(0, ["Al"])

    with pytest.raises(ValueError, match="Slider value must be between 1 and 10"):
        get_heuristics_for_slider(11, ["Al"])


def test_heuristics_invalid_type() -> None:
    with pytest.raises(TypeError, match="Slider value must be an integer"):
        get_heuristics_for_slider(5.5, ["Al"])  # type: ignore[arg-type]
