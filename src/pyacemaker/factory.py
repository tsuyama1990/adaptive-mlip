from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import PyAceConfig


# Default implementations for Cycle 01
class Cycle01Generator(BaseGenerator):
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """Placeholder generator for Cycle 01."""
        # Yield nothing or simple dummy atoms if needed, but for now empty
        # Real implementation will use config.structure
        return iter([])


class Cycle01Trainer(BaseTrainer):
    def train(self, training_data_path: str | Path) -> Any:
        """Placeholder trainer for Cycle 01."""
        return None


class Cycle01Engine(BaseEngine):
    def run(self, structure: Atoms | None, potential: Any) -> Any:
        """Placeholder engine for Cycle 01."""
        return None


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
            # For Cycle 02, we use DFTManager for Oracle if configured
            # config.dft is always present due to Pydantic model
            oracle = DFTManager(config.dft)

            return (
                Cycle01Generator(),
                oracle,
                Cycle01Trainer(),
                Cycle01Engine(),
            )
        except Exception as e:
            msg = f"Failed to create modules: {e}"
            raise RuntimeError(msg) from e
