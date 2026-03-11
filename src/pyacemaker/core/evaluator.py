import logging

from pyacemaker.domain_models.workflow import ActiveLearningThresholds

logger = logging.getLogger(__name__)


class TwoTierEvaluator:
    """
    Two-Tier Evaluator to differentiate thermal noise from true physical anomalies.
    Maintains a stateful tracking of max_gamma values across MD steps.
    """

    def __init__(self, thresholds: ActiveLearningThresholds) -> None:
        self.thresholds = thresholds
        self.consecutive_exceed_count = 0

    def evaluate(self, max_gamma: float) -> bool:
        """
        Evaluates whether an uncertainty (max_gamma) truly requires a halt.

        Args:
            max_gamma: The maximum uncertainty observed in the current step.

        Returns:
            bool: True if a true anomaly is detected and MD should halt, False otherwise.
        """
        if max_gamma > self.thresholds.threshold_call_dft:
            self.consecutive_exceed_count += 1
            if self.consecutive_exceed_count >= self.thresholds.smooth_steps:
                return True
            logger.warning("Thermal Noise Detected, Ignoring")
        else:
            self.consecutive_exceed_count = 0
        return False
