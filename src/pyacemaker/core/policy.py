from typing import Any

from pyacemaker.core.base import BasePolicy


class SafeBasePolicy(BasePolicy):
    def generate(self, **kwargs: Any) -> None:
        """
        Generates new candidates based on policy logic.
        """
        # Validate allowed kwargs
        allowed_args = {"n_candidates", "engine", "potential", "structure", "exploration_config"}
        unknown = set(kwargs.keys()) - allowed_args
        if unknown:
            raise ValueError(f"Unknown arguments passed to Policy.generate: {unknown}")

# Re-implement ColdStartPolicy and others that might have been overwritten or missing
class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)
        # Cold start logic stub

class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)

class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)

class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)

class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)

class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)

class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """
    def generate(self, **kwargs: Any) -> None:
        super().generate(**kwargs)
