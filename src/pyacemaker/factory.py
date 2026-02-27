from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.core.oracle import DFTManager
from pyacemaker.core.report import ReportGenerator
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.phonons import PhononCalculator


class ModuleFactory:
    """
    Factory for creating core modules based on configuration.
    """

    @staticmethod
    def create_modules(
        config: PyAceConfig,
    ) -> tuple[BaseGenerator, BaseOracle, BaseTrainer, BaseEngine, ActiveSetSelector, Validator]:
        """
        Creates instances of core modules based on the provided configuration.

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
        if not config.project_name:
            msg = "Project name is required for module initialization"
            raise ConfigError(msg)

        if config.distillation.enable_mace_distillation:
            return ModuleFactory._create_distillation_modules(config)

        return ModuleFactory._create_standard_modules(config)

    @staticmethod
    def _create_distillation_modules(
        config: PyAceConfig,
    ) -> tuple[BaseGenerator, BaseOracle, BaseTrainer, BaseEngine, ActiveSetSelector, Validator]:
        """Creates modules for MACE distillation workflow."""
        from pyacemaker.modules.mace_oracle import MaceOracle

        try:
            # Generator
            generator = StructureGenerator(config.structure)

            # Oracle
            mace_config = config.distillation.step3_mace_finetune
            # We need a place for MaceOracle model path.
            # It's in step3 config (base_model).

            oracle = MaceOracle(
                model_path=mace_config.base_model,
                device="cuda" if config.training.use_gpu else "cpu"
            )

            # Trainer: PacemakerTrainer (used in later steps)
            trainer = PacemakerTrainer(config.training)

            # Engine: LammpsEngine (used for MD)
            engine = LammpsEngine(config.md)

            # Active Set Selector
            active_set_selector = ActiveSetSelector()

            # Validator
            report_gen = ReportGenerator()
            phonon_calc = PhononCalculator(
                engine,
                config.validation.phonon_supercell,
                config.validation.phonon_displacement,
                config.validation.phonon_imaginary_tol,
            )
            elastic_calc = ElasticCalculator(
                engine,
                config.validation.elastic_strain,
                config.validation.elastic_steps,
            )
            validator = Validator(
                config.validation, phonon_calc, elastic_calc, report_gen
            )

            return (generator, oracle, trainer, engine, active_set_selector, validator)

        except Exception as e:
            msg = f"Failed to create distillation modules: {e}"
            raise RuntimeError(msg) from e

    @staticmethod
    def _create_standard_modules(
        config: PyAceConfig,
    ) -> tuple[BaseGenerator, BaseOracle, BaseTrainer, BaseEngine, ActiveSetSelector, Validator]:
        """Creates standard modules for the legacy workflow."""
        try:
            # Oracle
            oracle = DFTManager(config.dft)

            # Generator
            generator = StructureGenerator(config.structure)

            # Trainer
            trainer = PacemakerTrainer(config.training)

            # Engine
            engine = LammpsEngine(config.md)

            # Active Set Selector
            active_set_selector = ActiveSetSelector()

            # Validator
            report_gen = ReportGenerator()
            phonon_calc = PhononCalculator(
                engine,
                config.validation.phonon_supercell,
                config.validation.phonon_displacement,
                config.validation.phonon_imaginary_tol,
            )
            elastic_calc = ElasticCalculator(
                engine,
                config.validation.elastic_strain,
                config.validation.elastic_steps,
            )
            validator = Validator(
                config.validation, phonon_calc, elastic_calc, report_gen
            )

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
