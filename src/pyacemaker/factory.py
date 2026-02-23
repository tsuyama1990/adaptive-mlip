from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models import PyAceConfig


class ModuleFactory:
    """
    Factory for creating core modules based on configuration.
    Currently returns None placeholders for future cycles.
    """

    @staticmethod
    def create_modules(
        config: PyAceConfig,
    ) -> tuple[BaseGenerator | None, BaseOracle | None, BaseTrainer | None, BaseEngine | None]:
        # Future: Instantiate concrete classes based on config.dft.code, config.training.type, etc.
        _ = config  # Prevent unused argument warning for now
        return None, None, None, None
