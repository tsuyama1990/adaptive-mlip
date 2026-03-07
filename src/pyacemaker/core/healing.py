from abc import ABC, abstractmethod

from pyacemaker.domain_models import DFTConfig


class HealingStrategy(ABC):
    @abstractmethod
    def apply(self, config: DFTConfig) -> None:
        pass

class ReduceBetaStrategy(HealingStrategy):
    def __init__(self, factor: float) -> None:
        self.factor = factor
    def apply(self, config: DFTConfig) -> None:
        config.mixing_beta *= self.factor

class IncreaseSmearingStrategy(HealingStrategy):
    def __init__(self, factor: float) -> None:
        self.factor = factor
    def apply(self, config: DFTConfig) -> None:
        config.smearing_width *= self.factor

class UseCGDiagonalizationStrategy(HealingStrategy):
    def apply(self, config: DFTConfig) -> None:
        config.diagonalization = "cg"
