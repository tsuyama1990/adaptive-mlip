import math
from typing import TypedDict

import numpy as np


class ActiveLearningThresholdsDict(TypedDict, total=False):
    threshold_call_dft: float
    threshold_add_train: float


class LoopStrategyDict(TypedDict, total=False):
    thresholds: ActiveLearningThresholdsDict


class WorkflowHeuristicsDict(TypedDict, total=False):
    loop_strategy: LoopStrategyDict


class MDHeuristicsDict(TypedDict, total=False):
    timestep: float
    check_interval: int
    uncertainty_threshold: float


class DFTHeuristicsDict(TypedDict, total=False):
    encut: float
    smearing_type: str
    smearing_width: float


class PacemakerHeuristicsDict(TypedDict, total=False):
    learning_rate: float


class TrainingHeuristicsDict(TypedDict, total=False):
    pacemaker: PacemakerHeuristicsDict


class HeuristicConfigDict(TypedDict, total=False):
    md: MDHeuristicsDict
    dft: DFTHeuristicsDict
    workflow: WorkflowHeuristicsDict
    training: TrainingHeuristicsDict


def _scale_dft_threshold(slider: int) -> float:
    """
    Exponentially scaling function for DFT call threshold.
    Accuracy (10) results in rigorous small thresholds.
    """
    exponent = -0.012 * (slider**2) - 0.01 * slider - 0.67
    val = float(np.power(10.0, exponent))
    return float(max(0.001, min(0.5, val)))


def _scale_md_timestep(slider: int) -> float:
    """
    Scales MD integration timestep. Returns picoseconds (ps).
    Speed (1) -> 2.0 fs = 0.002 ps
    Accuracy (10) -> <= 0.5 fs = 0.0005 ps
    """
    val_fs = 2.38 * float(np.exp(-0.173 * slider))
    val_fs = max(0.1, min(3.0, val_fs))
    return float(val_fs * 0.001)


def _scale_check_interval(slider: int) -> int:
    """
    Frequency of uncertainty evaluation.
    Speed (1) -> 20 steps
    Accuracy (10) -> 1 step
    """
    val = 28.5 * float(np.exp(-0.33 * slider))
    interval = math.floor(val)
    return max(1, min(100, interval))


def _scale_encut(slider: int) -> float:
    """
    Scales the Quantum Espresso kinetic energy cutoff.
    Speed (1) -> 30 Ry
    Accuracy (10) -> 80 Ry
    """
    val = 30.55 - 1.166 * slider + 0.6111 * (slider**2)
    return float(max(20.0, min(150.0, val)))


def _scale_learning_rate(slider: int) -> float:
    """
    Scales the MLIP training learning rate.
    """
    val = 0.077 * float(np.exp(-0.434 * slider))
    return float(max(1e-6, min(0.1, val)))


def get_heuristics_for_slider(value: int, element_context: list[str]) -> HeuristicConfigDict:
    """
    Pure mathematical functional mapping from a 1-10 slider to exact physical parameters.
    """
    if not isinstance(value, int):
        msg = f"Slider value must be an integer. Got {type(value)}"
        raise TypeError(msg)

    if not (1 <= value <= 10):
        msg = f"Slider value must be between 1 and 10. Got {value}"
        raise ValueError(msg)

    dft_threshold = _scale_dft_threshold(value)
    md_timestep = _scale_md_timestep(value)
    check_interval = _scale_check_interval(value)
    encut = _scale_encut(value)
    learning_rate = _scale_learning_rate(value)

    # Contextual fallbacks
    smearing_type = "gaussian"
    smearing_width = 0.1
    if "Pt" in element_context:
        smearing_type = "mv"
        smearing_width = 0.02

    return {
        "md": {
            "timestep": md_timestep,
            "check_interval": check_interval,
            "uncertainty_threshold": dft_threshold,
        },
        "dft": {
            "encut": encut,
            "smearing_type": smearing_type,
            "smearing_width": smearing_width,
        },
        "workflow": {
            "loop_strategy": {
                "thresholds": {
                    "threshold_call_dft": dft_threshold,
                    "threshold_add_train": dft_threshold / 2.0,
                }
            }
        },
        "training": {
            "pacemaker": {
                "learning_rate": learning_rate,
            }
        },
    }
