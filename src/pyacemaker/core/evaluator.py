import logging
from typing import Any

logger = logging.getLogger(__name__)


class TwoTierEvaluator:
    def __init__(
        self, threshold_call_dft: float, threshold_add_train: float, smooth_steps: int,
        max_retries: int = 3, backoff_base: float = 0.1
    ) -> None:
        self.threshold_call_dft = threshold_call_dft
        self.threshold_add_train = threshold_add_train
        self.smooth_steps = smooth_steps
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.consecutive_exceedances = 0

    def _raise_extraction_error(self) -> None:
        msg = "Failed to extract max_g variable from LAMMPS."
        raise RuntimeError(msg)

    def evaluate(self, lmp: "Any") -> None:
        """
        Evaluate logic called by LAMMPS via fix python/invoke.
        """
        try:
            max_gamma = None
            import time
            for attempt in range(self.max_retries):
                try:
                    max_gamma = lmp.extract_variable("max_g")
                    break
                except Exception as err:
                    if attempt == self.max_retries - 1:
                        msg = "Failed to extract max_g variable after retries."
                        raise RuntimeError(msg) from err
                    time.sleep(self.backoff_base * (2 ** attempt)) # exponential backoff

            if max_gamma is None:
                self._raise_extraction_error()
                return  # Added strictly for static analysis typing, though `_raise_extraction_error` raises

            # Ensure mypy knows it's a float now
            mg = float(max_gamma)

            if mg > self.threshold_call_dft:
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
                # Trigger halt
                lmp.command("variable trigger_halt string true")
        except Exception as e:
            logger.exception("TwoTierEvaluator encountered an error")
            msg = "TwoTierEvaluator encountered an error"
            raise RuntimeError(msg) from e


# To use via fix python/invoke, we need a module-level wrapper that instantiates and calls evaluate.
# We will generate a script dynamically in lammps_generator.py to initialize it with correct config.
