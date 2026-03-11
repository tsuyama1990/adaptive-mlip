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
    def validate_structure(structure: Atoms) -> None:  # noqa: C901
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
            # Script injection and sanitization check:
            if not isinstance(s, str) or not s.isalpha() or len(s) > 2:
                msg = f"Chemical symbol contains invalid characters or types: {s}"
                raise ValueError(msg)

            if s not in atomic_numbers:
                raise ValueError(ERR_VAL_STRUCT_UNKNOWN_SYM.format(symbol=s))
            if atomic_numbers[s] == 0:
                raise ValueError(ERR_VAL_STRUCT_DUMMY_ELEM.format(symbol=s))

    @staticmethod
    def validate_potential(potential: str | Path) -> Path:
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

    def _calculate_eos(self, structure: Atoms, potential_path: Path) -> str:
        """
        Calculates Equation of State (EOS) by varying volume and computing energy.
        Returns a base64 encoded plot of the EOS fit.
        """
        import base64
        import io

        import matplotlib.pyplot as plt
        import numpy as np
        from ase.eos import EquationOfState

        volumes = []
        energies = []

        engine = self.elastic_calc.engine
        base_cell = structure.get_cell()  # type: ignore[no-untyped-call]

        # Strain from -5% to +5% volume
        strains = np.linspace(-0.05, 0.05, 7)
        for eps in strains:
            atoms = structure.copy()  # type: ignore[no-untyped-call]
            # Isotropic volume scaling
            scale_factor = (1.0 + eps) ** (1.0 / 3.0)
            cell = base_cell * scale_factor
            atoms.set_cell(cell, scale_atoms=True)

            result = engine.compute_static_properties(atoms, potential_path)
            volumes.append(atoms.get_volume())
            energies.append(result.energy)

        # Fit EOS (Birch-Murnaghan)
        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")  # type: ignore[no-untyped-call]
            v0, e0, B = eos.fit()  # type: ignore[no-untyped-call]

            # Plot
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            eos.plot(ax=ax)  # type: ignore[no-untyped-call]

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return ""

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

        # Equation of State (EOS) Check
        eos_plot = self._calculate_eos(relaxed_structure, potential_path)

        result = ValidationResult(
            phonon_stable=phonon_stable,
            elastic_stable=elastic_stable,
            c_ij=c_ij,
            bulk_modulus=B,
            plots={"phonon": phonon_plot, "elastic": elastic_plot, "eos": eos_plot},
            report_path=str(output_path),
        )

        # Generate Report
        html = self.report_gen.generate(result)
        self.report_gen.save(output_path, html)

        return result
