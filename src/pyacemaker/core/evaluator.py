import logging

logger = logging.getLogger(__name__)

class TwoTierEvaluator:
    def __init__(self, threshold_call_dft: float, threshold_add_train: float, smooth_steps: int) -> None:
        self.threshold_call_dft = threshold_call_dft
        self.threshold_add_train = threshold_add_train
        self.smooth_steps = smooth_steps
        self.consecutive_exceedances = 0

    def evaluate(self, lmp) -> None:
        """
        Evaluate logic called by LAMMPS via fix python/invoke.
        """
        try:
            # Extract system max_gamma (which should be calculated by pace/mace pair style)
            # For this mock/real implementation, we assume max_gamma is available as a variable
            max_gamma = lmp.extract_variable("max_g")

            if max_gamma > self.threshold_call_dft:
                self.consecutive_exceedances += 1
                logger.debug(f"TwoTierEvaluator: max_gamma ({max_gamma:.4f}) > threshold ({self.threshold_call_dft}). Consecutive: {self.consecutive_exceedances}/{self.smooth_steps}")
            else:
                if self.consecutive_exceedances > 0:
                    logger.debug(f"TwoTierEvaluator: max_gamma ({max_gamma:.4f}) <= threshold. Resetting consecutive exceedances (was {self.consecutive_exceedances}).")
                self.consecutive_exceedances = 0

            if self.consecutive_exceedances >= self.smooth_steps:
                logger.info(f"TwoTierEvaluator: Threshold exceeded for {self.smooth_steps} consecutive steps. Triggering halt.")
                # Trigger halt
                lmp.command("variable trigger_halt string true")
        except Exception:
            logger.exception("TwoTierEvaluator encountered an error")
            raise

# To use via fix python/invoke, we need a module-level wrapper that instantiates and calls evaluate.
# We will generate a script dynamically in lammps_generator.py to initialize it with correct config.
