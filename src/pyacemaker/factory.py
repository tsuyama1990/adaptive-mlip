from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.container import Container
from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig


class ModuleFactory:
    """
    Factory for creating core modules based on configuration.
    delegates to DI Container.
    """

    @staticmethod
    def create_modules(
        config: PyAceConfig,
    ) -> tuple[BaseGenerator, BaseOracle, BaseTrainer, BaseEngine, ActiveSetSelector, Validator]:
        """
        Creates instances of core modules using the Container.
        Kept for backward compatibility if needed, or used as a helper.
        """
        if not config.project_name:
            msg = "Project name is required for module initialization"
            raise ConfigError(msg)

        try:
            container = Container.create(config)
        except Exception as e:
            msg = f"Failed to create modules: {e}"
            raise RuntimeError(msg) from e

        return (
            container.generator,
            container.oracle,
            container.trainer,
            container.engine,
            container.active_set_selector,
            container.validator,
        )

    @staticmethod
    def create_container(config: PyAceConfig) -> Container:
        """
        Creates and returns the DI Container.
        """
        if not config.project_name:
            msg = "Project name is required for module initialization"
            raise ConfigError(msg)
        try:
            return Container.create(config)
        except Exception as e:
            msg = f"Failed to create container: {e}"
            raise RuntimeError(msg) from e
