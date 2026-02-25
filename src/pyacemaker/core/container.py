from dataclasses import dataclass
from typing import Any

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.core.oracle import DFTManager
from pyacemaker.core.report import ReportGenerator
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.phonons import PhononCalculator


@dataclass
class Container:
    """
    Dependency Injection Container.
    Holds instances of core components.
    """
    config: PyAceConfig
    generator: BaseGenerator
    oracle: BaseOracle
    trainer: BaseTrainer
    engine: BaseEngine
    active_set_selector: ActiveSetSelector
    validator: Validator

    @classmethod
    def create(cls, config: PyAceConfig) -> "Container":
        """
        Factory method to create the container and its dependencies.
        """
        # Engine (SimulationEngine + PropertyCalculator + RelaxationEngine)
        engine = LammpsEngine(config.md)

        # Oracle
        oracle = DFTManager(config.dft)

        # Generator
        generator = StructureGenerator(config.structure)

        # Trainer
        trainer = PacemakerTrainer(config.training)

        # Active Set Selector
        active_set_selector = ActiveSetSelector()

        # Validator components
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

        return cls(
            config=config,
            generator=generator,
            oracle=oracle,
            trainer=trainer,
            engine=engine,
            active_set_selector=active_set_selector,
            validator=validator,
        )
