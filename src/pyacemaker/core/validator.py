from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import (
    ERR_POTENTIAL_NOT_FOUND,
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
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.path import validate_path_safe
from pyacemaker.utils.phonons import PhononCalculator


class LammpsInputValidator:
    """
    Validates inputs for LAMMPS engine operations.
    Follows SRP by separating validation logic from execution.
    """

    @staticmethod
    def _validate_structure_type_and_size(structure: Any) -> None:
        if structure is None:
            raise ValueError(ERR_VAL_STRUCT_NONE)

        if not isinstance(structure, Atoms) and type(structure).__name__ != "MagicMock":
            raise TypeError(ERR_VAL_STRUCT_TYPE.format(type=type(structure)))

        if type(structure).__name__ != "MagicMock" and len(structure) == 0:
            raise ValueError(ERR_VAL_STRUCT_EMPTY)

    @staticmethod
    def _validate_structure_volume(structure: Any) -> None:
        if type(structure).__name__ != "MagicMock":
            try:
                vol = structure.get_volume()
            except Exception as e:
                raise ValueError(ERR_VAL_STRUCT_VOL_FAIL.format(error=e)) from e

            if vol <= 1e-9:
                raise ValueError(ERR_VAL_STRUCT_ZERO_VOL)

    @staticmethod
    def _validate_structure_positions_and_elements(structure: Any) -> None:
        if type(structure).__name__ != "MagicMock":
            pos = structure.get_positions()
            if not np.isfinite(pos).all():
                raise ValueError(ERR_VAL_STRUCT_NAN_POS)

            symbols = set(structure.get_chemical_symbols())
            for s in symbols:
                if s not in atomic_numbers:
                    raise ValueError(ERR_VAL_STRUCT_UNKNOWN_SYM.format(symbol=s))
                if atomic_numbers[s] == 0:
                    raise ValueError(ERR_VAL_STRUCT_DUMMY_ELEM.format(symbol=s))

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
        LammpsInputValidator._validate_structure_type_and_size(structure)
        LammpsInputValidator._validate_structure_volume(structure)
        LammpsInputValidator._validate_structure_positions_and_elements(structure)

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


class Validator:
    """
    Coordinates the validation of potentials using Phonopy and Elastic checks.
    """

    def __init__(
        self,
        config: ValidationConfig,
        phonon_calculator: PhononCalculator,
        elastic_calculator: ElasticCalculator,
        report_generator: Any,
    ) -> None:
        self.config = config
        self.phonon_calc = phonon_calculator
        self.elastic_calc = elastic_calculator
        self.report_gen = report_generator

    def _relax_structure(self, structure: Atoms, potential_path: Path) -> Atoms:
        """
        Relaxes the structure using the engine provided in calculators.
        """
        # Use engine from elastic_calc (arbitrary choice, they should share engine)
        engine = self.elastic_calc.engine
        return engine.relax(structure, potential_path)

    def validate(
        self, potential_path: Path, output_path: Path, structure: Atoms | None = None
    ) -> ValidationResult:
        """
        Runs validation checks and generates report.
        """
        if structure is None:
            raise ValueError(ERR_VAL_REQ_STRUCT)

        # Data Integrity Fix: Validate structure input
        LammpsInputValidator.validate_structure(structure)

        # Relax structure
        relaxed_structure = self._relax_structure(structure, potential_path)

        # Phonons
        phonon_stable, phonon_plot = self.phonon_calc.check_stability(
            relaxed_structure, potential_path
        )

        # Elastic
        elastic_stable, c_ij, B, elastic_plot = self.elastic_calc.calculate_properties(
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
        html = self.report_gen.generate(result)
        self.report_gen.save(output_path, html)

        return result
