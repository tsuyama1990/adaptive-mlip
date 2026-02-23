from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.core.oracle import DFTManager
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models import PyAceConfig


class ModuleFactory:
    """
    Factory for creating core modules based on configuration.
    """

    @staticmethod
    def create_modules(
        config: PyAceConfig,
    ) -> tuple[BaseGenerator, BaseOracle, BaseTrainer, BaseEngine]:
        """
        Creates instances of core modules.

        Raises:
            ConfigError: If configuration is invalid.
            RuntimeError: If module creation fails.
        """
        # Validate configuration before module creation
        if not config.project_name:
            msg = "Project name is required for module initialization"
            raise ConfigError(msg)

        try:
            # Oracle
            oracle = DFTManager(config.dft)

            # Generator
            generator = StructureGenerator(config.structure)

            # Trainer
            trainer = PacemakerTrainer(config.training)

            # Engine
            engine = LammpsEngine(config.md)

        except Exception as e:
            msg = f"Failed to create modules: {e}"
            raise RuntimeError(msg) from e

        return (
            generator,
            oracle,
            trainer,
            engine,
        )
