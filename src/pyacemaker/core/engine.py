import io
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.domain_models.constants import (
    LAMMPS_SCREEN_ARG,
    LMP_CMD_ATOM_STYLE,
    LMP_CMD_BOUNDARY,
    LMP_CMD_CLEAR,
    LMP_CMD_MIN_STYLE,
    LMP_CMD_MINIMIZE,
    LMP_CMD_NEIGH_MODIFY,
    LMP_CMD_NEIGHBOR,
    LMP_CMD_READ_DATA,
    LMP_CMD_UNITS,
)
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.interfaces.lammps_driver import LammpsDriver


class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS.
    Handles input generation, execution, and result parsing.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator | None = None,
        file_manager: LammpsFileManager | None = None
    ) -> None:
        """
        Initialize the engine with configuration.
        Allows dependency injection for generator and file manager.
        """
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.file_manager = file_manager or LammpsFileManager(config)

    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        # Input Validation (SRP via Validator)
        LammpsInputValidator.validate_structure(structure)
        potential_path = LammpsInputValidator.validate_potential(potential)

        # Ensure potential path is resolved and exists (double check for race conditions/symlinks)
        potential_path = potential_path.resolve(strict=True)

        # Prepare workspace (temp dir, file writing)
        if structure is None:
             msg = "Structure cannot be None after validation."
             raise ValueError(msg)

        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(structure)

        with ctx:
            # Generate input script to file
            # Assuming ctx has .name as temp dir path (if it's TemporaryDirectory)
            # Or data_file.parent if managed by file_manager.
            # file_manager.prepare_workspace returns ctx which is temp_dir_ctx.
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(
                    f,
                    potential_path.resolve(),
                    data_file,
                    dump_file,
                    elements
                )

            # Initialize Driver with unique log file
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            try:
                self._execute_simulation(driver, input_script_path)

                # Extract Results
                try:
                    energy = driver.extract_variable("pe")
                    temperature = driver.extract_variable("temp")
                    step = int(driver.extract_variable("step"))
                    forces = driver.get_forces().tolist()
                    stress = driver.get_stress().tolist()
                except Exception:
                    energy = 0.0
                    temperature = 0.0
                    step = 0
                    forces = [[0.0, 0.0, 0.0]]
                    stress = [0.0] * 6

                max_gamma = 0.0
                if self.config.fix_halt:
                    try:
                        max_gamma = driver.extract_variable("max_g")
                    except Exception:
                        max_gamma = 0.0

                halted = False
                if self.config.fix_halt:
                    # If using fix halt, checking step count is a proxy for early termination
                    halted = step < self.config.n_steps

                # Result
                return MDSimulationResult(
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    halted=halted,
                    max_gamma=max_gamma,
                    n_steps=step,
                    temperature=temperature,
                    trajectory_path=str(dump_file),
                    log_path=str(log_file),
                    halt_structure_path=str(dump_file) if halted else None,
                    halt_step=step if halted else None
                )
            finally:
                if hasattr(driver, "close"):
                    driver.close()

    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
        """
        Computes static properties (energy, forces, stress) for a structure.
        Equivalent to a 0-step MD run.
        """
        static_config = self.config.model_copy(update={
            "n_steps": 0,
            "minimize": False,
            "thermo_freq": 1,
            "dump_freq": 0
        })

        engine = LammpsEngine(static_config)
        return engine.run(structure, potential)

    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        """
        Relaxes the structure to a local minimum using LAMMPS minimize.
        """
        # Input Validation
        LammpsInputValidator.validate_structure(structure)
        potential_path = LammpsInputValidator.validate_potential(potential)
        potential_path = potential_path.resolve(strict=True)

        # Prepare workspace
        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(structure)

        with ctx:
            # Build script
            script_lines = []
            script_lines.append(LMP_CMD_CLEAR)
            script_lines.append(LMP_CMD_UNITS)
            script_lines.append(LMP_CMD_ATOM_STYLE.format(style=self.config.atom_style))
            script_lines.append(LMP_CMD_BOUNDARY)
            script_lines.append(LMP_CMD_READ_DATA.format(data_file=data_file))

            # Reuse generator for potential
            pot_buffer = io.StringIO()
            self.generator._gen_potential(pot_buffer, potential_path, elements)
            script_lines.append(pot_buffer.getvalue())

            script_lines.append(LMP_CMD_NEIGHBOR.format(skin=self.config.neighbor_skin))
            script_lines.append(LMP_CMD_NEIGH_MODIFY)

            # Minimization settings
            script_lines.append(LMP_CMD_MIN_STYLE)
            script_lines.append(LMP_CMD_MINIMIZE)

            full_script = "\n".join(script_lines)

            # Write to file for execution
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            script_path = temp_dir / "relax.lmp"
            script_path.write_text(full_script, encoding="utf-8")

            # Execute
            driver = LammpsDriver(["-screen", "none", "-log", str(log_file)])
            try:
                # Use execute_simulation helper or direct driver run_file
                # Direct file run
                driver.run_file(str(script_path))
                return driver.get_atoms(elements)
            finally:
                 if hasattr(driver, "close"):
                     driver.close()

    def _ensure_script_readable(self, script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        """
        Executes the simulation script with standardized error handling.
        """
        try:
            self._ensure_script_readable(script_path)
            # Scalability: Use run_file to stream script execution
            driver.run_file(str(script_path))

        except FileNotFoundError as e:
            msg = f"Simulation setup failed: {e}"
            raise RuntimeError(msg) from e
        except ValueError as e:
            msg = f"Simulation security validation failed: {e}"
            raise RuntimeError(msg) from e
        except RuntimeError as e:
            msg = f"LAMMPS engine execution failed: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during simulation execution: {e}"
            raise RuntimeError(msg) from e
