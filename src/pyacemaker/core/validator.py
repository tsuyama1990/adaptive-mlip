from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read

from pyacemaker.core.report import ReportGenerator
from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.phonons import PhononCalculator


class LammpsValidator:
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
            ValueError: If structure is invalid or empty.
            TypeError: If input is not an ASE Atoms object.
        """
        if structure is None:
            msg = "Structure must be provided."
            raise ValueError(msg)

        if not isinstance(structure, Atoms):
            msg = f"Expected ASE Atoms object, got {type(structure)}."
            raise TypeError(msg)

        if len(structure) == 0:
            msg = "Structure contains no atoms."
            raise ValueError(msg)

    @staticmethod
    def validate_potential(potential: Any) -> Path:
        """
        Validates the potential path.

        Args:
            potential: Path to potential file (str or Path).

        Returns:
            Validated Path object.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If input is invalid.
        """
        if potential is None:
            msg = "Potential path must be provided."
            raise ValueError(msg)

        path = Path(potential)
        if not path.exists():
            msg = f"Potential file not found: {path}"
            raise FileNotFoundError(msg)

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
        report_generator: ReportGenerator,
    ) -> None:
        self.config = config
        self.phonon_calc = phonon_calculator
        self.elastic_calc = elastic_calculator
        self.report_gen = report_generator

    def _relax_structure(self, structure: Atoms, potential_path: Path) -> Atoms:
        """
        Relaxes the structure using the engine provided in calculators.
        Assuming both calculators use the same engine instance.
        """
        # Use engine from elastic_calc (arbitrary choice, they should share engine)
        engine = self.elastic_calc.engine

        # Create a config for minimization
        # We need to access engine.config to copy/modify it.
        # But BaseEngine interface doesn't expose config.
        # However, we know it's LammpsEngine in this context.
        # If we cannot access config, we can rely on default behavior of 'run' if configured?
        # No, 'run' runs MD by default.
        # We need a dedicated relax method on Engine or hack config.
        # Since I am the implementer, I can assume LammpsEngine has 'config'.
        # Or better, I added `compute_static_properties` which does static.
        # I need `relax_structure` on Engine.
        # But adding another method to BaseEngine...
        # Let's check `LammpsEngine.run`. If `minimize=True` in config, it minimizes.
        # I can create a temporary engine with minimize=True.

        if hasattr(engine, "config"):
             relax_config = engine.config.model_copy(update={
                 "minimize": True,
                 "n_steps": 0, # Just minimization
                 "thermo_freq": 10
             })
             # Create new engine of same type
             RelaxEngine = type(engine)
             relax_engine = RelaxEngine(relax_config)

             result = relax_engine.run(structure, potential_path)
             if result.trajectory_path:
                 # Read last frame
                 # ase.io.read returns Atoms or list of Atoms
                 relaxed = read(result.trajectory_path, index=-1)
                 if isinstance(relaxed, list):
                     return relaxed[-1]
                 return relaxed # type: ignore

        return structure

    def validate(
        self, potential_path: Path, output_path: Path, structure: Atoms | None = None
    ) -> ValidationResult:
        """
        Runs validation checks and generates report.
        """
        if structure is None:
            # TODO: In future, generate a structure based on elements?
            # For now, require it.
            # But Orchestrator might call it without structure if we didn't update it.
            # I'll raise error.
            msg = "Validation requires a structure."
            raise ValueError(msg)

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
