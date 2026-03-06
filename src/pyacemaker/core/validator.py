from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import (
    ERR_POTENTIAL_NOT_FOUND,
    ERR_VAL_POT_NONE,
    ERR_VAL_POT_NOT_FILE,
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
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.path import validate_path_safe
from pyacemaker.utils.phonons import PhononCalculator


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
        if not path.exists():
            raise FileNotFoundError(ERR_POTENTIAL_NOT_FOUND.format(path=path))

        if not path.is_file():
            raise ValueError(ERR_VAL_POT_NOT_FILE.format(path=path))

        return path


class PhononValidator:
    """Validator for phonon stability."""
    def __init__(self, calculator: PhononCalculator) -> None:
        self.calculator = calculator

    def validate(self, structure: Atoms, potential_path: Path) -> tuple[bool, Path]:
        return self.calculator.check_stability(structure, potential_path)

class ElasticValidator:
    """Validator for elastic stability."""
    def __init__(self, calculator: ElasticCalculator) -> None:
        self.calculator = calculator

    def validate(self, structure: Atoms, potential_path: Path) -> tuple[bool, list[list[float]], float, Path]:
        return self.calculator.calculate_properties(structure, potential_path)

class ReportValidator:
    """Generates validation reports."""
    def __init__(self, report_generator: Any) -> None:
        self.report_gen = report_generator

    def generate(self, result: ValidationResult, output_path: Path) -> None:
        html = self.report_gen.generate(result)
        self.report_gen.save(output_path, html)

class StructureRelaxer:
    """Relaxes structures for validation."""
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def relax(self, structure: Atoms, potential_path: Path) -> Atoms:
        return self.engine.relax(structure, potential_path)

class Validator:
    """
    Coordinates the validation of potentials using Phonopy and Elastic checks.
    Uses composition of specialized validators to adhere to Single Responsibility Principle.
    """

    def __init__(
        self,
        config: ValidationConfig,
        phonon_validator: PhononValidator,
        elastic_validator: ElasticValidator,
        report_validator: ReportValidator,
        relaxer: StructureRelaxer,
    ) -> None:
        self.config = config
        self.phonon_validator = phonon_validator
        self.elastic_validator = elastic_validator
        self.report_validator = report_validator
        self.relaxer = relaxer

    def validate(
        self, potential_path: Path, output_path: Path, structure: Atoms
    ) -> ValidationResult:
        """
        Runs validation checks and generates report.
        """
        # Data Integrity Fix: Validate structure input
        LammpsInputValidator.validate_structure(structure)

        # Relax structure
        relaxed_structure = self.relaxer.relax(structure, potential_path)

        # Phonons
        phonon_stable, phonon_plot = self.phonon_validator.validate(
            relaxed_structure, potential_path
        )

        # Elastic
        elastic_stable, c_ij, B, elastic_plot = self.elastic_validator.validate(
            relaxed_structure, potential_path
        )

        result = ValidationResult(
            phonon_stable=phonon_stable,
            elastic_stable=elastic_stable,
            c_ij=c_ij,
            bulk_modulus=B,
            plots={"phonon": phonon_plot, "elastic": elastic_plot},
            report_path=str(output_path),
        )

        # Generate Report
        from pyacemaker.utils.path import validate_path_safe
        output_path = validate_path_safe(output_path)
        self.report_validator.generate(result, output_path)

        return result
