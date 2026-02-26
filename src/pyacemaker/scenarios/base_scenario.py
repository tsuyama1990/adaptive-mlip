from abc import ABC, abstractmethod

from pyacemaker.domain_models.config import PyAceConfig


class BaseScenario(ABC):
    """
    Abstract base class for all scenarios.
    Scenarios encapsulate specific scientific workflows or user stories.
    """

    def __init__(self, config: PyAceConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        """Returns the name of the scenario."""
        if self.config.scenario:
            return self.config.scenario.name
        return "unknown"

    @abstractmethod
    def run(self) -> None:
        """Executes the scenario logic."""
