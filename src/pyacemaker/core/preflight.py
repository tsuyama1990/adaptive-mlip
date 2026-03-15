import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
        except (ValueError, TypeError, KeyError) as e:
            report.errors.append(
                DiagnosticMessage(
                    node_id="INITIAL_STRUCTURE",
                    severity=Severity.ERROR,
                    description=f"Configuration error while generating initial structure: {e}",
                    suggestion="Verify the chemical symbols and lattice parameters.",
                )
            )
            return
        except StopIteration:
            report.errors.append(
                DiagnosticMessage(
                    node_id="INITIAL_STRUCTURE",
                    severity=Severity.ERROR,
                    description="Generator failed to produce an initial structure.",
                    suggestion="Check exploration policy constraints.",
                )
            )
            return
        except Exception as e:
            report.errors.append(
                DiagnosticMessage(
                    node_id="INITIAL_STRUCTURE",
                    severity=Severity.ERROR,
                    description=f"Unexpected error generating initial structure: {e}",
                    suggestion="Review generator configurations.",
                )
            )
            return

        import numpy as np
        from scipy.spatial.distance import pdist, squareform

        positions = atoms.get_positions()  # type: ignore[no-untyped-call]

        if len(positions) == 0:
            report.errors.append(DiagnosticMessage(
                node_id="INITIAL_STRUCTURE", severity=Severity.ERROR,
                description="Initial structure contains 0 atoms.",
                suggestion="Check the input coordinates or file parsing logic."
            ))
            return

        if len(positions) == 1:
            # Check edge case single atom logic
            if not np.isfinite(positions).all():
                report.errors.append(DiagnosticMessage(
                    node_id="INITIAL_STRUCTURE", severity=Severity.ERROR,
                    description="Single-atom structure contains invalid (NaN or Inf) coordinates.",
                    suggestion="Check atomic coordinates."
                ))
            return  # No collisions possible

        if not np.isfinite(positions).all():
            report.errors.append(DiagnosticMessage(
                node_id="INITIAL_STRUCTURE", severity=Severity.ERROR,
                description="Structure contains invalid (NaN or Inf) coordinates.",
                suggestion="Check atomic coordinates."
            ))
            return

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

        self._check_volume(atoms, report)

    def _check_volume(self, atoms: "Any", report: DiagnosticReport) -> None:
        """Helper to check cell volume density."""
        volume = atoms.get_volume()
        if volume > 0.0 and len(atoms) / volume > 0.5:
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

    def _validate_executables(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        required_executables = []
        if config.dft.code.lower() in ("qe", "quantum_espresso"):
            required_executables.append("pw.x")

        if config.training.potential_type.lower() == "mace":
            required_executables.append("mace_run_train")

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

    def _validate_pseudopotentials(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        from pyacemaker.utils.path import validate_path_safe

        if config.dft.pseudopotentials:
            for element, filename in config.dft.pseudopotentials.items():
                try:
                    safe_filename = validate_path_safe(Path(str(filename)))
                    if not safe_filename.exists():
                         report.errors.append(
                            DiagnosticMessage(
                                node_id="DFT_CONFIG",
                                severity=Severity.ERROR,
                                description=f"Pseudopotential file not found for element {element}: {safe_filename}",
                                suggestion="Ensure the UPF file exists in the correct directory.",
                            )
                        )
                except ValueError as e:
                     report.errors.append(
                        DiagnosticMessage(
                            node_id="DFT_CONFIG",
                            severity=Severity.ERROR,
                            description=f"Invalid pseudopotential path for element {element}: {e}",
                            suggestion="Provide a secure path within the project directory.",
                        )
                    )

    def _validate_models(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        from pyacemaker.utils.path import validate_path_safe

        foundation_model_path = getattr(config.training, "foundation_model_path", None)
        if foundation_model_path:
            try:
                p = validate_path_safe(Path(str(foundation_model_path)))
                if not p.exists():
                    report.errors.append(
                        DiagnosticMessage(
                            node_id="MACE_TRAINING",
                            severity=Severity.ERROR,
                            description=f"Required MACE model file not found at path: {p}",
                            suggestion="Verify the path to the pre-trained MACE model.",
                        )
                    )
            except ValueError as e:
                report.errors.append(
                    DiagnosticMessage(
                        node_id="MACE_TRAINING",
                        severity=Severity.ERROR,
                        description=f"Invalid MACE model path: {e}",
                        suggestion="Provide a valid, secure path within the project directory.",
                    )
                )

    def validate(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        self._validate_executables(config, report)
        self._validate_pseudopotentials(config, report)
        self._validate_models(config, report)


class LammpsSyntaxValidator(BaseValidator):
    """
    Validates LAMMPS script syntax by spinning up a lightweight isolated Python subprocess.
    """
    def _execute_dry_run(self, lammps_script_file: Path, runner_py: Path, report: DiagnosticReport) -> None:
        runner_code = """
from lammps import lammps
import sys

lmp = lammps(cmdargs=["-screen", "none"])
try:
    script_path = sys.argv[1]
    with open(script_path, 'r') as f:
        script = f.read()
    for line in script.splitlines():
        if line.strip():
            lmp.command(line.strip())
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
"""
        runner_py.write_text(runner_code)

        safe_env = os.environ.copy()
        if "PYTHONPATH" in safe_env:
            del safe_env["PYTHONPATH"]

        try:
            from pyacemaker.utils.path import validate_path_safe
            safe_runner_py = validate_path_safe(runner_py)
            safe_script_file = validate_path_safe(lammps_script_file)
        except ImportError as e:
            report.errors.append(DiagnosticMessage(
                node_id="SYSTEM", severity=Severity.ERROR,
                description=f"Missing internal utility dependency: {e}",
                suggestion="Ensure pyacemaker installation is complete."
            ))
            return

        proc = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(safe_runner_py.resolve()),
                str(safe_script_file.resolve())
            ],
            capture_output=True,
            text=True,
            env=safe_env,
            check=False,
            shell=False,
        )

        if proc.returncode != 0:
            err_out = proc.stderr.strip()
            err_lines = [line for line in err_out.splitlines() if "ERROR" in line]
            if err_lines:
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

    def _prepare_files(self, atoms: "Any", td_path: Path, config: PyAceConfig, report: DiagnosticReport) -> Path | None:
        import stat
        data_file = td_path / "structure.data"
        try:
            from ase.io import write
        except ImportError as e:
            report.errors.append(DiagnosticMessage(
                node_id="SYSTEM", severity=Severity.ERROR,
                description=f"Missing ASE dependency: {e}",
                suggestion="Install ase via 'uv add ase'"
            ))
            return None

        try:
            from pyacemaker.utils.validation import validate_structure
            validate_structure(atoms)
        except ImportError as e:
            report.errors.append(DiagnosticMessage(
                node_id="SYSTEM", severity=Severity.ERROR,
                description=f"Missing internal dependency: {e}",
                suggestion="Ensure pyacemaker is installed fully."
            ))
            return None
        except Exception as e:
             report.errors.append(DiagnosticMessage(
                node_id="INITIAL_STRUCTURE", severity=Severity.ERROR,
                description=f"Initial structure validation failed: {e}",
                suggestion="Check atomic structure configuration."
            ))
             return None

        style = getattr(config.md.atom_style, "value", config.md.atom_style) if config.md.atom_style else "atomic"
        write(data_file, atoms, format="lammps-data", atom_style=style)
        data_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

        potential_path = td_path / "potential.yace"
        potential_path.touch()
        potential_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        from pyacemaker.core.lammps_generator import LammpsScriptGenerator
        elements = list(set(atoms.get_chemical_symbols()))
        lammps_gen = LammpsScriptGenerator(config.md)
        dump_file = td_path / "dump.xyz"

        import io
        buffer = io.StringIO()
        lammps_gen.write_script(buffer, potential_path, data_file, dump_file, elements)

        script_content = buffer.getvalue()
        lines = script_content.splitlines()
        modified_lines = []
        for line in lines:
            if line.startswith("run "):
                modified_lines.append("run 0")
            elif line.startswith(("fix py_halt", "python eval_wrapper")):
                pass
            else:
                modified_lines.append(line)

        lammps_script_file = td_path / "script.in"
        if not self._validate_and_write_script(modified_lines, lammps_script_file, report):
            return None

        lammps_script_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return lammps_script_file

    def _validate_and_write_script(self, lines: list[str], path: Path, report: DiagnosticReport) -> bool:
        """Validates and writes the script to disk."""
        try:
            from pyacemaker.utils.validation import validate_lammps_command
            for line in lines:
                if line.strip():
                    validate_lammps_command(line.strip())
        except ImportError as e:
            report.errors.append(DiagnosticMessage(
                node_id="SYSTEM", severity=Severity.ERROR,
                description=f"Missing internal dependency: {e}",
                suggestion="Ensure pyacemaker is installed fully."
            ))
            return False
        except ValueError as e:
            report.errors.append(DiagnosticMessage(
                node_id="LAMMPS_SCRIPT", severity=Severity.ERROR,
                description=f"LAMMPS command validation failed: {e}",
                suggestion="Ensure the script does not contain shell injections or dangerous commands."
            ))
            return False

        path.touch()
        path.write_text("\n".join(lines))
        return True

    def validate(self, config: PyAceConfig, report: DiagnosticReport) -> None:
        try:
            from pyacemaker.factory import ModuleFactory
            generator, _, _, _, _, _ = ModuleFactory.create_modules(config)
            atoms = next(generator.generate(1))
        except Exception:
            return

        import stat
        td = tempfile.mkdtemp()
        td_path = Path(td)
        try:
            lammps_script_file = self._prepare_files(atoms, td_path, config, report)
            if not lammps_script_file:
                return

            runner_py = td_path / "runner.py"
            runner_py.touch()
            runner_py.chmod(stat.S_IRUSR | stat.S_IWUSR)

            self._execute_dry_run(lammps_script_file, runner_py, report)
        finally:
            try:
                shutil.rmtree(td_path)
            except Exception as e:
                try:
                    from pyacemaker.domain_models.logging import LoggingConfig
                    from pyacemaker.logger import setup_logger
                    logger = setup_logger(config=LoggingConfig(), project_name="api_gateway")
                    logger.warning(f"Failed to cleanup preflight temp directory {td_path}: {e}")
                except Exception as inner_e:
                    import sys
                    sys.stderr.write(f"Failed to cleanup temp directory and logger failed: {inner_e}\n")


class PreflightManager:
    """Orchestrates preflight diagnostic validations."""

    def __init__(self, validators: list[BaseValidator] | None = None) -> None:
        if validators is None:
            self.validators = [
                StructuralValidator(),
                DependencyValidator(),
                LammpsSyntaxValidator(),
            ]
        else:
            self.validators = validators

    def run(self, config: PyAceConfig) -> DiagnosticReport:
        report = DiagnosticReport()
        for validator in self.validators:
            try:
                validator.validate(config, report)
            except ImportError as e:
                report.errors.append(
                    DiagnosticMessage(
                        node_id="PREFLIGHT",
                        severity=Severity.ERROR,
                        description=f"Dependency missing in {validator.__class__.__name__}: {e}",
                        suggestion="Check your python environment dependencies.",
                    )
                )
            except OSError as e:
                report.errors.append(
                    DiagnosticMessage(
                        node_id="PREFLIGHT",
                        severity=Severity.ERROR,
                        description=f"File system error in {validator.__class__.__name__}: {e}",
                        suggestion="Check permissions and disk space.",
                    )
                )
            except Exception as e:
                from pyacemaker.domain_models.logging import LoggingConfig
                from pyacemaker.logger import setup_logger
                logger = setup_logger(config=LoggingConfig(), project_name="api_gateway")
                logger.exception(f"Validator {validator.__class__.__name__} crashed unexpectedly.")
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
