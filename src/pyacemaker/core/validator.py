import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write

from pyacemaker.core.report import ReportGenerator
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult
from pyacemaker.interfaces.lammps_driver import LammpsDriver
from pyacemaker.utils.elastic import ElasticCalculator
from pyacemaker.utils.phonons import PhononCalculator
from pyacemaker.utils.structure import get_species_order

logger = logging.getLogger(__name__)


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
    Orchestrates the validation of a potential using Phonons and Elastic constants.
    """

    def __init__(
        self,
        potential_path: Path,
        validation_config: ValidationConfig,
        structure: Atoms,
        md_config: MDConfig | None = None,
        report_generator: ReportGenerator | None = None,
    ) -> None:
        """
        Args:
            potential_path: Path to the potential file (.yace).
            validation_config: Configuration for validation (thresholds, supercells).
            structure: The structure to validate.
            md_config: Optional MD configuration (needed for hybrid potential parameters).
            report_generator: Optional ReportGenerator instance (dependency injection).
        """
        self.potential_path = potential_path
        self.config = validation_config
        self.structure = structure
        self.md_config = md_config
        self.driver = LammpsDriver(["-screen", "none", "-log", "none"])
        self.report_generator = report_generator or ReportGenerator()

    def validate(self) -> ValidationResult:
        """
        Runs the full validation suite.
        """
        logger.info("Starting validation for potential: %s", self.potential_path)

        # 1. Relaxation
        logger.info("Relaxing structure...")
        relaxed_structure = self._relax_structure(self.structure)

        # 2. Phonons
        logger.info("Running phonon calculation...")
        phonon_calc = PhononCalculator(relaxed_structure, self.config.phonon_supercell)

        # Define force function for phonopy
        def force_fn(atoms: Atoms) -> list[list[float]]:
            forces = self._calculate_forces(atoms)
            # Ensure it's a list[list[float]]
            if isinstance(forces, np.ndarray):
                return forces.tolist()  # type: ignore[no-any-return]
            return list(forces)

        phonon_calc.calculate_forces(force_fn)
        phonon_stable, imaginary_freqs = phonon_calc.check_stability(
            tolerance=self.config.imaginary_frequency_tolerance
        )

        # Generate plots
        plots = {}
        try:
            plots["band_structure"] = phonon_calc.get_band_structure_plot()
            plots["dos"] = phonon_calc.get_dos_plot()
        except Exception as e:
            logger.warning("Failed to generate phonon plots: %s", e)

        # 3. Elastic
        logger.info("Running elastic calculation...")
        elastic_calc = ElasticCalculator(relaxed_structure, strain=self.config.elastic_strain)

        # Define stress function
        def stress_fn(atoms: Atoms) -> np.ndarray:
            return self._calculate_stress(atoms)

        Cij, B, G = elastic_calc.calculate(stress_fn)

        # Basic check: B > 0, G > 0.
        # Born criteria for cubic: C11-C12 > 0, C11+2C12 > 0, C44 > 0
        # For general, checking eigenvalues of Cij matrix is better but complex.
        # We stick to simple moduli check + manual inspection of Cij.
        elastic_stable = ElasticCalculator.check_stability(B, G)

        result = ValidationResult(
            phonon_stable=phonon_stable,
            elastic_stable=elastic_stable,
            imaginary_frequencies=imaginary_freqs,
            elastic_tensor=Cij,
            bulk_modulus=B,
            shear_modulus=G,
            plots=plots
        )

        # Generate report
        # We save to current working directory or potential directory?
        # Default to "validation_report.html" in CWD for now.
        try:
            self.report_generator.save(result, "validation_report.html")
        except Exception as e:
            logger.warning("Failed to save validation report: %s", e)

        logger.info("Validation completed. Phonon: %s, Elastic: %s", phonon_stable, elastic_stable)
        return result

    def _relax_structure(self, atoms: Atoms) -> Atoms:
        """
        Minimizes the structure using LAMMPS.
        """
        # We run a minimization 'run 0' style but with 'minimize'.
        # We need to extract the final structure.
        # LammpsDriver.get_atoms() does this.

        self._run_lammps(atoms, minimize=True)
        elements = get_species_order(atoms)
        return self.driver.get_atoms(elements)

    def _calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """Calculates forces for a given structure."""
        self._run_lammps(atoms, minimize=False)
        return self.driver.get_forces()

    def _calculate_stress(self, atoms: Atoms) -> np.ndarray:
        """Calculates stress for a given structure."""
        self._run_lammps(atoms, minimize=False)
        return self.driver.get_stress()

    def _run_lammps(self, atoms: Atoms, minimize: bool = False) -> None:
        """
        Sets up and runs LAMMPS for the given atoms.
        """
        self.driver.lmp.command("clear")
        self.driver.lmp.command("units metal")
        self.driver.lmp.command("atom_style atomic")
        self.driver.lmp.command("boundary p p p")

        # Write data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lmp", delete=True) as tmp:
            elements = get_species_order(atoms)
            # Use 'atomic' style strictly here as we set atom_style atomic above
            # If MDConfig has charge, we might need to adjust.
            # Assuming 'atomic' for validation for now unless hybrid needs charge?
            # ZBL uses atomic numbers, usually 'atomic' is fine.
            write(tmp.name, atoms, format="lammps-data", specorder=elements, atom_style="atomic")
            self.driver.lmp.command(f"read_data {tmp.name}")

        # Potential setup
        self._setup_potential(elements)

        # Compute stress if needed (for get_stress)
        self.driver.lmp.command("thermo_style custom step temp pe press pxx pyy pzz pyz pxz pxy")

        if minimize:
            # Minimize
            self.driver.lmp.command("minimize 1.0e-4 1.0e-6 100 1000")
        else:
            # Run 0 to get forces/stress
            self.driver.lmp.command("run 0")

    def _setup_potential(self, elements: list[str]) -> None:
        """Generates potential commands."""
        species_str = " ".join(elements)
        # Use str(Path) as LammpsDriver runs python strings
        pot_path = str(self.potential_path.resolve())

        # Hybrid check
        is_hybrid = False
        params = None
        if self.md_config and self.md_config.hybrid_potential:
            is_hybrid = True
            params = self.md_config.hybrid_params

        if is_hybrid and params:
            self.driver.lmp.command(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}")
            self.driver.lmp.command(f"pair_coeff * * pace {pot_path} {species_str}")

            n_types = len(elements)
            for i in range(n_types):
                el_i = elements[i]
                z_i = atomic_numbers[el_i]
                for j in range(i, n_types):
                    el_j = elements[j]
                    z_j = atomic_numbers[el_j]
                    self.driver.lmp.command(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}")
        else:
            self.driver.lmp.command("pair_style pace")
            self.driver.lmp.command(f"pair_coeff * * pace {pot_path} {species_str}")
