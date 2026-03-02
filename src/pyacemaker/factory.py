from collections.abc import Callable
from typing import Any

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.core.oracle import DFTManager
from pyacemaker.core.report import ReportGenerator
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.core.validator import ValidationContext, Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.phonons import PhononCalculator


class DIContainer:
    """Dependency Injection Container with lifecycle management."""

    def __init__(self) -> None:
        self._transient_providers: dict[str, Callable[[PyAceConfig], Any]] = {}
        self._singleton_providers: dict[str, Callable[[PyAceConfig], Any]] = {}
        self._singleton_instances: dict[str, Any] = {}

    def register_transient(self, name: str, provider: Callable[[PyAceConfig], Any]) -> None:
        """Register a dependency that creates a new instance every time it is resolved."""
        self._transient_providers[name] = provider

    def register_singleton(self, name: str, provider: Callable[[PyAceConfig], Any]) -> None:
        """Register a dependency that creates a single instance and caches it."""
        self._singleton_providers[name] = provider

    def resolve(self, name: str, config: PyAceConfig) -> Any:
        """Resolve a dependency by name."""
        if name in self._transient_providers:
            return self._transient_providers[name](config)
        if name in self._singleton_providers:
            if name not in self._singleton_instances:
                self._singleton_instances[name] = self._singleton_providers[name](config)
            return self._singleton_instances[name]
        msg = f"No provider registered for {name}"
        raise KeyError(msg)


_default_container = DIContainer()
_default_container.register_transient("oracle", lambda c: DFTManager(c.dft))
_default_container.register_transient("generator", lambda c: StructureGenerator(c.structure))
_default_container.register_transient("trainer", lambda c: PacemakerTrainer(c.training))
_default_container.register_singleton("engine", lambda c: LammpsEngine(c.md))
_default_container.register_transient("active_set_selector", lambda c: ActiveSetSelector())
_default_container.register_singleton("report_gen", lambda c: ReportGenerator())
_default_container.register_transient(
    "phonon_calc",
    lambda c: PhononCalculator(
        _default_container.resolve("engine", c),
        c.validation.phonon_supercell,
        c.validation.phonon_displacement,
        c.validation.phonon_imaginary_tol,
    ),
)
_default_container.register_transient(
    "elastic_calc",
    lambda c: ElasticCalculator(
        _default_container.resolve("engine", c),
        c.validation.elastic_strain,
        c.validation.elastic_steps,
    ),
)


def _create_validator(c: PyAceConfig) -> Validator:
    context = ValidationContext(
        config=c.validation,
        phonon_calculator=_default_container.resolve("phonon_calc", c),
        elastic_calculator=_default_container.resolve("elastic_calc", c),
        report_generator=_default_container.resolve("report_gen", c),
    )
    return Validator(context)


_default_container.register_transient("validator", _create_validator)


class ModuleFactory:
    """
    Factory for creating core modules based on configuration.
    """

    container: DIContainer = _default_container

    @classmethod
    def create_modules(
        cls,
        config: PyAceConfig,
    ) -> tuple[BaseGenerator, BaseOracle, BaseTrainer, BaseEngine, ActiveSetSelector, Validator]:
        """
        Creates instances of core modules based on the provided configuration.

        This method acts as a dependency injection root, instantiating concrete implementations
        of the core abstract base classes (Generator, Oracle, Trainer, Engine, ActiveSetSelector).

        Args:
            config: A validated PyAceConfig object containing all necessary settings.

        Returns:
            A tuple containing initialized instances of:
                - BaseGenerator (e.g., StructureGenerator)
                - BaseOracle (e.g., DFTManager)
                - BaseTrainer (e.g., PacemakerTrainer)
                - BaseEngine (e.g., LammpsEngine)
                - ActiveSetSelector
                - Validator

        Raises:
            ConfigError: If configuration is invalid or missing required fields.
            RuntimeError: If any module fails to initialize (e.g., missing dependencies).
        """
        # Validate configuration before module creation
        if not config.project_name:
            msg = "Project name is required for module initialization"
            raise ConfigError(msg)

        try:
            oracle = cls.container.resolve("oracle", config)
            generator = cls.container.resolve("generator", config)
            trainer = cls.container.resolve("trainer", config)
            engine = cls.container.resolve("engine", config)
            active_set_selector = cls.container.resolve("active_set_selector", config)
            validator = cls.container.resolve("validator", config)

        except Exception as e:
            msg = f"Failed to create modules: {e}"
            raise RuntimeError(msg) from e

        return (
            generator,
            oracle,
            trainer,
            engine,
            active_set_selector,
            validator,
        )
