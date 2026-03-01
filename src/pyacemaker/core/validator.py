from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import (
    ERR_VAL_POT_NONE,
    ERR_VAL_POT_NOT_FILE,
    ERR_VAL_REQ_STRUCT,
    ERR_VAL_STRUCT_DUMMY_ELEM,
    ERR_VAL_STRUCT_EMPTY,
    ERR_VAL_STRUCT_NAN_POS,
    ERR_VAL_STRUCT_NONE,
    ERR_VAL_STRUCT_TYPE,
    ERR_VAL_STRUCT_UNKNOWN_SYM,
    ERR_VAL_STRUCT_VOL_FAIL,
    ERR_VAL_STRUCT_ZERO_VOL,
)
from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult
from pyacemaker.utils.path import validate_path_safe


class LammpsInputValidator:
    """
    Validates inputs for LAMMPS engine operations.
    Follows SRP by separating validation logic from execution.
    """

    @staticmethod
    def validate_structure(structure: Any) -> None:
        """
        Validates the atomic structure.

        Args:
            structure: Input structure object.

        Raises:
            ValueError: If structure is invalid, empty, or contains unknown elements.
            TypeError: If input is not an ASE Atoms object.
        """
        if structure is None:
            raise ValueError(ERR_VAL_STRUCT_NONE)

        if not isinstance(structure, Atoms):
            raise TypeError(ERR_VAL_STRUCT_TYPE.format(type=type(structure)))

        if len(structure) == 0:
            raise ValueError(ERR_VAL_STRUCT_EMPTY)

        # Validate structure physical properties
        try:
            vol = structure.get_volume()  # type: ignore[no-untyped-call]
        except Exception as e:
            # get_volume might fail if no cell is set
            raise ValueError(ERR_VAL_STRUCT_VOL_FAIL.format(error=e)) from e

        if vol <= 1e-9:
            raise ValueError(ERR_VAL_STRUCT_ZERO_VOL)

        # Validate positions are numeric and finite
        pos = structure.get_positions()  # type: ignore[no-untyped-call]
        if not np.isfinite(pos).all():
            raise ValueError(ERR_VAL_STRUCT_NAN_POS)

        # Validate elements against atomic_numbers
        symbols = set(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]
        for s in symbols:
            if s not in atomic_numbers:
                raise ValueError(ERR_VAL_STRUCT_UNKNOWN_SYM.format(symbol=s))
            if atomic_numbers[s] == 0:
                raise ValueError(ERR_VAL_STRUCT_DUMMY_ELEM.format(symbol=s))

    @staticmethod
    def validate_potential(potential: Any) -> Path:
        """
        Validates the potential path.
        Ensures path exists, is a file, and is within allowed directories using secure validation.

        Args:
            potential: Path to potential file (str or Path).

        Returns:
            Validated Path object.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If input is invalid or path is insecure.
        """
        if potential is None:
            raise ValueError(ERR_VAL_POT_NONE)

        # Convert to Path but do not resolve yet; validate_path_safe handles resolution checks
        p = Path(potential)

        # Use centralized secure validator
        path = validate_path_safe(p)

        # Additional checks for existence (validate_path_safe ensures safety, not existence)
        # Note: validate_path_safe uses strict resolution for existing files to avoid TOCTOU
        # But if the file was created between calls or we just want to ensure it is still a file

        # It's already resolved securely, just check type
        if not path.is_file():
            raise ValueError(ERR_VAL_POT_NOT_FILE.format(path=path))

        return path



class BasePhononCalculator(ABC):
    @abstractmethod
    def check_stability(self, structure: Atoms, potential_path: Path) -> tuple[bool, str | None]:
        ...

class BaseElasticCalculator(ABC):
    @property
    @abstractmethod
    def engine(self) -> Any:
        ...

    @abstractmethod
    def calculate_properties(self, structure: Atoms, potential_path: Path) -> tuple[bool, dict[str, float], float, str | None]:
        ...


class StructureRelaxer:
    def __init__(self, engine: Any):
        self.engine = engine

    def relax(self, structure: Atoms, potential_path: Path) -> Atoms:
        return self.engine.relax(structure, potential_path)

class PhononValidator:
    def __init__(self, calculator: BasePhononCalculator):
        self.calculator = calculator

    def validate(self, structure: Atoms, potential_path: Path) -> tuple[bool, str | None]:
        return self.calculator.check_stability(structure, potential_path)

class ElasticValidator:
    def __init__(self, calculator: BaseElasticCalculator):
        self.calculator = calculator

    def validate(self, structure: Atoms, potential_path: Path) -> tuple[bool, dict[str, float], float, str | None]:
        return self.calculator.calculate_properties(structure, potential_path)

class ValidationCoordinator:
    """
    Orchestrates the validation pipeline by delegating to specific validators.
    """
    def __init__(
        self,
        config: ValidationConfig,
        phonon_validator: PhononValidator,
        elastic_validator: ElasticValidator,
        structure_relaxer: StructureRelaxer,
        report_generator: Any,
    ) -> None:
        self.config = config
        self.phonon_validator = phonon_validator
        self.elastic_validator = elastic_validator
        self.structure_relaxer = structure_relaxer
        self.report_gen = report_generator

    def validate(
        self, potential_path: Path, output_path: Path, structure: Atoms | None = None
    ) -> ValidationResult:
        if structure is None:
            raise ValueError(ERR_VAL_REQ_STRUCT)

        LammpsInputValidator.validate_structure(structure)
        relaxed_structure = self.structure_relaxer.relax(structure, potential_path)

        phonon_stable, phonon_plot = self.phonon_validator.validate(relaxed_structure, potential_path)
        elastic_stable, c_ij, B, elastic_plot = self.elastic_validator.validate(relaxed_structure, potential_path)

        result = ValidationResult(
            phonon_stable=phonon_stable,
            elastic_stable=elastic_stable,
            c_ij=c_ij,
            bulk_modulus=B,
            plots={"phonon": phonon_plot, "elastic": elastic_plot},
            report_path=str(output_path),
        )

        html = self.report_gen.generate(result)
        self.report_gen.save(output_path, html)

        return result


# Maintain backwards compatibility for tests
class Validator(ValidationCoordinator):
    def __init__(
        self,
        config: ValidationConfig,
        phonon_calculator: BasePhononCalculator,
        elastic_calculator: BaseElasticCalculator,
        report_generator: Any,
    ) -> None:
        super().__init__(
            config=config,
            phonon_validator=PhononValidator(phonon_calculator),
            elastic_validator=ElasticValidator(elastic_calculator),
            structure_relaxer=StructureRelaxer(elastic_calculator.engine),
            report_generator=report_generator
        )

    def _relax_structure(self, structure: Atoms, potential_path: Path) -> Atoms:
        return self.structure_relaxer.relax(structure, potential_path)
