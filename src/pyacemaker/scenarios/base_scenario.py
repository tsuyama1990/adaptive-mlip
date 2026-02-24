import logging
from abc import ABC, abstractmethod

from pyacemaker.domain_models.config import PyAceConfig

logger = logging.getLogger(__name__)


class BaseScenario(ABC):
    """
    Abstract base class for all scenarios.

    A Scenario encapsulates a specific workflow logic (e.g., Grand Challenge,
    custom deployment) that differs from the standard active learning loop.
    """

    def __init__(self, config: PyAceConfig) -> None:
        self.config = config
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        """
        Executes the scenario workflow.
        Should handle its own error reporting and cleanup.
        """
