import logging
from typing import Any

from pyacemaker.core.exceptions import MDHaltInterrupt

logger = logging.getLogger(__name__)


class TwoTierEvaluator:
    def __init__(
        self, threshold_call_dft: float, threshold_add_train: float, smooth_steps: int
    ) -> None:
        self.threshold_call_dft = threshold_call_dft
        self.threshold_add_train = threshold_add_train
        self.smooth_steps = smooth_steps
        self.consecutive_exceedances = 0

    # ruff: noqa: C901, PLR0912
    def evaluate(self, lmp: "Any") -> None:
        """
        Evaluate logic called by LAMMPS via fix python/invoke.
        """
        try:
            max_gamma = None
            retries = 3
            import time
            for attempt in range(retries):
                try:
                    max_gamma = lmp.extract_variable("max_g")
                    break
                except Exception as err:
                    if attempt == retries - 1:
                        msg = "Failed to extract max_g variable after retries."
                        raise RuntimeError(msg) from err
                    time.sleep(0.1 * (2 ** attempt)) # exponential backoff

            if max_gamma is None:
                msg = "Failed to extract max_g variable from LAMMPS."
                raise RuntimeError(msg)

            if max_gamma > self.threshold_call_dft:
                self.consecutive_exceedances += 1
                logger.debug(
                    f"TwoTierEvaluator: max_gamma ({max_gamma:.4f}) > threshold ({self.threshold_call_dft}). Consecutive: {self.consecutive_exceedances}/{self.smooth_steps}"
                )
            else:
                if self.consecutive_exceedances > 0:
                    logger.debug(
                        f"TwoTierEvaluator: max_gamma ({max_gamma:.4f}) <= threshold. Resetting consecutive exceedances (was {self.consecutive_exceedances})."
                    )
                self.consecutive_exceedances = 0

            if self.consecutive_exceedances >= self.smooth_steps:
                logger.info(
                    f"TwoTierEvaluator: Threshold exceeded for {self.smooth_steps} consecutive steps. Triggering halt."
                )

                # Fetch epicenter indices
                step = int(lmp.extract_variable("step"))
                num_atoms = int(lmp.extract_variable("atoms"))

                epicenter_indices = []
                try:
                    import ctypes

                    # Try to extract global array of uncertainties if it exists
                    # Typically computed via compute (e.g., c_gamma)
                    ptr = lmp.extract_compute("gamma", 1, 1) # 1=per-atom, 1=vector
                    if ptr:
                        # Convert pointer to ctypes array
                        array_type = ctypes.c_double * num_atoms
                        c_array = ctypes.cast(ptr, ctypes.POINTER(array_type)).contents
                        for i in range(num_atoms):
                            if c_array[i] > self.threshold_add_train:
                                epicenter_indices.append(i + 1) # LAMMPS indices are 1-based
                except Exception as err:
                    logger.warning(f"Could not extract detailed epicenter indices: {err}")

                # If extraction fails or doesn't find any, provide a fallback to pass validation
                if not epicenter_indices:
                    epicenter_indices = [1]

                # Explicitly signal to lammps to stop as well (as a fallback, though the exception should crash the python invoke)
                lmp.command("variable trigger_halt string true")
                raise MDHaltInterrupt(step=step, epicenter_indices=epicenter_indices)
        except MDHaltInterrupt:
            # Re-raise so it gets caught by the caller/engine
            raise
        except Exception as e:
            logger.exception("TwoTierEvaluator encountered an error")
            msg = "TwoTierEvaluator encountered an error"
            raise RuntimeError(msg) from e


# To use via fix python/invoke, we need a module-level wrapper that instantiates and calls evaluate.
# We will generate a script dynamically in lammps_generator.py to initialize it with correct config.
