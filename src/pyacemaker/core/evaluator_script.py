"""
Static execution script for TwoTierEvaluator.
This is meant to be invoked from LAMMPS via `python/invoke`.
It reads thresholds dynamically from a JSON file in the same directory.
"""

import json
import sys
from pathlib import Path

# Add src to sys.path to allow imports from pyacemaker inside LAMMPS
src_path = Path(__file__).parent.parent.parent.resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyacemaker.core.engine import TwoTierEvaluator  # noqa: E402
from pyacemaker.domain_models.workflow import ActiveLearningThresholds  # noqa: E402

# Global evaluator instance initialized once per LAMMPS process
eval_uncertainty = None


def init_evaluator(lmp: object) -> None:
    global eval_uncertainty  # noqa: PLW0603
    if eval_uncertainty is not None:
        return

    # Extract the configuration path securely from the LAMMPS variable
    try:
        config_path = str(lmp.extract_variable("evaluator_config_path", None, 0))  # type: ignore
    except Exception as e:
        msg = f"Failed to extract configuration path from LAMMPS: {e}"
        raise RuntimeError(msg) from e

    p = Path(config_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Perform strict Pydantic model validation on loaded JSON structure
    # to entirely prevent arbitrary execution or injection bugs
    thresholds = ActiveLearningThresholds.model_validate(data, strict=True)
    eval_uncertainty = TwoTierEvaluator(thresholds)


def eval_wrapper(lmp: object) -> None:
    """Wrapper to safely call eval_uncertainty from LAMMPS."""
    if eval_uncertainty is None:
        msg = "Evaluator not initialized. Ensure init_evaluator is called."
        raise RuntimeError(msg)
    eval_uncertainty(lmp)


# For lammps python/invoke it expects a function directly.
# We will dynamically overwrite eval_uncertainty at runtime when creating the workspace,
# or we can simply have lammps initialize it. For simplicity we will have the generator
# load the state directly.
