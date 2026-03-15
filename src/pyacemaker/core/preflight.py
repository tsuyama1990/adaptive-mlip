import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.preflight import DiagnosticMessage, DiagnosticReport, Severity


class BaseValidator(ABC):
    """Abstract base class for all preflight validators."""

    @abstractmethod
    def validate(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        """Executes the validation logic and updates the report."""


class StructuralValidator(BaseValidator):
    """
    Validates the physical structure (atomic collisions, cell volume) mathematically.
    Uses vectorized NumPy and SciPy operations to execute in milliseconds.
    """

    def validate(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        # We need an atoms object. Let's initialize the generator to get the initial structure.
        from pyacemaker.factory import ModuleFactory

        try:
            generator, _, _, _, _, _ = ModuleFactory.create_modules(config)
            atoms_iter = generator.generate(1)
            atoms = next(atoms_iter)
        except Exception as e:
            report.errors.append(
                DiagnosticMessage(
                    node_id="INITIAL_STRUCTURE",
                    severity=Severity.ERROR,
                    description=f"Failed to generate initial structure: {e}",
                    suggestion="Verify the chemical symbols and lattice parameters.",
                )
            )
            return

        positions = atoms.get_positions()
        if len(positions) < 2:
            return  # No collisions possible with < 2 atoms

        # Use pdist to calculate pairwise distances
        distances = pdist(positions)

        # We need the indices of colliding atoms if distance < 0.5 A
        # To get indices, we can use squareform or just find indices directly
        sq_dists = squareform(distances)
        np.fill_diagonal(sq_dists, np.inf)  # ignore self-distance

        collision_indices = np.where(sq_dists < 0.5)

        # Prevent reporting A-B and B-A twice
        reported = set()
        for i, j in zip(collision_indices[0], collision_indices[1], strict=False):
            i_idx, j_idx = (j, i) if i > j else (i, j)
            if (i_idx, j_idx) not in reported:
                reported.add((i_idx, j_idx))
                dist = sq_dists[i_idx, j_idx]
                report.errors.append(
                    DiagnosticMessage(
                        node_id="INITIAL_STRUCTURE",
                        severity=Severity.ERROR,
                        description=f"Atomic collision detected between Atom Index {i_idx} and Atom Index {j_idx}. Distance: {dist:.1f}A.",
                        suggestion="Please relax the initial structure using a classical potential before initiating the active learning loop, or manually edit the input coordinates to remove the overlap.",
                    )
                )

        # Cell volume check (simple heuristic: density check)
        volume = atoms.get_volume()
        if volume > 0 and len(atoms) / volume > 0.5:
             report.warnings.append(
                 DiagnosticMessage(
                    node_id="INITIAL_STRUCTURE",
                    severity=Severity.WARNING,
                    description=f"High atomic density detected: {len(atoms)/volume:.2f} atoms/A^3.",
                    suggestion="Verify that the simulation box volume is sufficient and lattice parameters are correct.",
                 )
             )


class DependencyValidator(BaseValidator):
    """
    Validates external dependencies (executables, models, pseudopotentials).
    """

    def validate(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        # Check required executables based on config
        required_executables = []
        if config.dft.code.lower() in ("qe", "quantum_espresso"):
            required_executables.append("pw.x")

        if config.training.potential_type.lower() == "mace":
            required_executables.append("mace_run_train") # Assuming pace_train or mace_run_train

        for exe in required_executables:
            if shutil.which(exe) is None:
                report.errors.append(
                    DiagnosticMessage(
                        node_id="SYSTEM_ENVIRONMENT",
                        severity=Severity.ERROR,
                        description=f"Required executable '{exe}' not found in PATH.",
                        suggestion=f"Install {exe} or ensure it is accessible in the system PATH.",
                    )
                )

        # Check explicit paths
        # Pseudopotentials
        if config.dft.pseudopotentials:
            for element, filename in config.dft.pseudopotentials.items():
                if not Path(filename).exists():
                     report.errors.append(
                        DiagnosticMessage(
                            node_id="DFT_CONFIG",
                            severity=Severity.ERROR,
                            description=f"Pseudopotential file not found for element {element}: {filename}",
                            suggestion="Ensure the UPF file exists in the correct directory.",
                        )
                    )

        # MACE model (advanced setting check)
        # Note: in config, advanced_settings might be mixed, we look for model paths if specified
        # If testing requires explicit path
        # Let's inspect config for any explicit path.
        # Typically MACE model is in TieredOracle or similar. For PyAceConfig, it might be in advanced settings or training.
        # However, looking at the SPEC: "user specifies an explicit path to a pre-trained MACE foundation model ... Check os.path.exists()"
        # The specific path may be passed.
        # We will parse `config` for any `.model` file.
        # The prompt says: "The user sets a visual node referencing a specific .model ... file"

        # Let's look for model paths in training config or scenario config advanced settings
        if getattr(config.training, "foundation_model_path", None):
            p = Path(config.training.foundation_model_path)
            if not p.exists():
                report.errors.append(
                    DiagnosticMessage(
                        node_id="MACE_TRAINING",
                        severity=Severity.ERROR,
                        description=f"Required MACE model file not found at path: {p}",
                        suggestion="Verify the path to the pre-trained MACE model.",
                    )
                )


class LammpsSyntaxValidator(BaseValidator):
    """
    Validates LAMMPS script syntax by spinning up a lightweight isolated Python subprocess.
    """

    def validate(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        from pyacemaker.factory import ModuleFactory
        from pyacemaker.core.lammps_generator import LammpsScriptGenerator

        try:
            generator, _, _, _, _, _ = ModuleFactory.create_modules(config)
            atoms = next(generator.generate(1))
        except Exception:
            return  # Will be caught by StructuralValidator

        elements = list(set(atoms.get_chemical_symbols()))
        lammps_gen = LammpsScriptGenerator(config.md)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # Generate a data file to load
            data_file = td_path / "structure.data"
            from ase.io import write
            write(data_file, atoms, format="lammps-data", atom_style=config.md.atom_style.value)

            potential_path = td_path / "potential.yace"
            potential_path.touch() # Dummy file for testing syntax
            dump_file = td_path / "dump.xyz"

            import io
            buffer = io.StringIO()
            lammps_gen.write_script(buffer, potential_path, data_file, dump_file, elements)

            script_content = buffer.getvalue()

            # Override run N to run 0 for syntax check
            # We want to replace the `run N` with `run 0`.
            lines = script_content.splitlines()
            modified_lines = []
            for line in lines:
                # If there's an intentional bad command injected by test, we keep it to let lammps crash
                if line.startswith("run "):
                    modified_lines.append("run 0")
                elif line.startswith("fix py_halt"):
                    # For preflight, skip Python fixes to avoid environment complications in subprocess
                    # if the user hasn't correctly set PYTHONPATH for the subprocess.
                    pass
                elif line.startswith("python eval_wrapper"):
                    pass
                else:
                    modified_lines.append(line)

            modified_script = "\n".join(modified_lines)

            # Write python runner script
            runner_py = td_path / "runner.py"
            runner_code = f"""
from lammps import lammps
import sys

lmp = lammps(cmdargs=["-screen", "none"])
script = {modified_script!r}
try:
    for line in script.splitlines():
        if line.strip():
            lmp.command(line.strip())
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
"""
            runner_py.write_text(runner_code)

            env = os.environ.copy()
            if 'PYTHONPATH' not in env:
                env['PYTHONPATH'] = 'src'

            proc = subprocess.run(  # noqa: S603
                [sys.executable, str(runner_py)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            if proc.returncode != 0:
                err_out = proc.stderr.strip()
                # Try to extract standard LAMMPS ERROR line
                err_lines = [line for line in err_out.splitlines() if "ERROR" in line]
                if err_lines:
                    # Clean up file/line reference
                    clean_err = err_lines[0].split("(")[0].strip()
                    desc = clean_err.replace("ERROR: ", "")
                else:
                    desc = "Unknown LAMMPS error"

                report.errors.append(
                    DiagnosticMessage(
                        node_id="ACTIVE_LEARNING_LOOP",
                        severity=Severity.ERROR,
                        description=desc,
                        suggestion="Check the LAMMPS configuration parameters for syntax issues.",
                    )
                )


class PreflightManager:
    """Orchestrates preflight diagnostic validations."""

    def __init__(self) -> None:
        self.validators: list[BaseValidator] = [
            StructuralValidator(),
            DependencyValidator(),
            LammpsSyntaxValidator(),
        ]

    def run(self, config: PyAceConfig) -> DiagnosticReport:
        report = DiagnosticReport()
        for validator in self.validators:
            try:
                validator.validate(config, report)
            except Exception as e:
                report.errors.append(
                    DiagnosticMessage(
                        node_id="PREFLIGHT",
                        severity=Severity.ERROR,
                        description=f"Validator {validator.__class__.__name__} crashed: {e}",
                        suggestion="Internal preflight error. Please contact support.",
                    )
                )

            # Fail fast if critical errors accumulated
            if report.errors:
                break

        if not report.errors and not report.warnings:
            report.info.append(
                DiagnosticMessage(
                    node_id="SYSTEM",
                    severity=Severity.INFO,
                    description="Preflight validation completed successfully.",
                    suggestion="Simulation is ready to dispatch.",
                )
            )

        return report
