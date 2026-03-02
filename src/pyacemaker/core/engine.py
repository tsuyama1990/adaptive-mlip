import tempfile
import time
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.domain_models.constants import (
    ERR_SIM_EXEC_FAIL,
    ERR_SIM_SECURITY_FAIL,
    ERR_SIM_SETUP_FAIL,
    ERR_SIM_UNEXPECTED,
    ERR_STRUCTURE_NONE,
    LAMMPS_SCREEN_ARG,
)
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.interfaces.lammps_driver import LammpsDriver


class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS, natively supporting Master-Slave Inversion via read_restart.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator | None = None,
        file_manager: LammpsFileManager | None = None,
    ) -> None:
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.file_manager = file_manager or LammpsFileManager(config)
        self._restart_file_path: str | None = None

    def _prepare_simulation_env(
        self, structure: Atoms | None, potential: Any, use_restart: bool = False
    ) -> tuple[Any, Path, Path, Path, list[str], Path]:
        if structure is None and not use_restart:
            raise ValueError(ERR_STRUCTURE_NONE)

        if structure is not None:
            LammpsInputValidator.validate_structure(structure)

        potential_path = LammpsInputValidator.validate_potential(potential)
        potential_path = potential_path.resolve(strict=True)

        if use_restart and self._restart_file_path:
            # If resuming, we don't need a data file from atoms.
            # We just need a dummy workspace.
            ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(
                Atoms("H")
            )
        else:
            ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(
                structure if structure is not None else Atoms("H")
            )

        return ctx, data_file, dump_file, log_file, elements, potential_path

    def _ensure_script_readable(self, script_path: Path) -> None:
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    def _execute_simulation(self, driver: LammpsDriver, script_path: Path) -> None:
        try:
            self._ensure_script_readable(script_path)
            driver.run_file(str(script_path))
        except FileNotFoundError as e:
            raise RuntimeError(ERR_SIM_SETUP_FAIL.format(error=e)) from e
        except ValueError as e:
            raise RuntimeError(ERR_SIM_SECURITY_FAIL.format(error=e)) from e
        except RuntimeError as e:
            raise RuntimeError(ERR_SIM_EXEC_FAIL.format(error=e)) from e
        except Exception as e:
            raise RuntimeError(ERR_SIM_UNEXPECTED.format(error=e)) from e

    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs the MD simulation. If a restart file is cached, resumes seamlessly.
        """
        use_restart = self._restart_file_path is not None
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential, use_restart)
        )

        with ctx:
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"
            restart_out_file = temp_dir / "checkpoint.restart"

            with input_script_path.open("w") as f:
                if use_restart and self._restart_file_path:
                    self.generator.write_resume_script(
                        f,
                        potential_path,
                        Path(self._restart_file_path),
                        restart_out_file,
                        dump_file,
                        elements,
                    )
                else:
                    self.generator.write_script(f, potential_path, data_file, dump_file, elements)
                    # Tell LAMMPS to write a restart file periodically
                    f.write(f"restart 1000 {restart_out_file}\n")

            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            try:
                self._execute_simulation(driver, input_script_path)

                # Save the latest restart file to an isolated path so it survives tempdir cleanup
                if restart_out_file.exists():
                    safe_restart_dir = (
                        Path(tempfile.gettempdir()) / f"lammps_restarts_{int(time.time())}"
                    )
                    safe_restart_dir.mkdir(parents=True, exist_ok=True)
                    safe_restart = safe_restart_dir / "latest.restart"
                    # We copy manually since Python 3.8+ shutil can be slow, but here read/write is fine.
                    safe_restart.write_bytes(restart_out_file.read_bytes())
                    self._restart_file_path = str(safe_restart)

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
                    halted = step < self.config.n_steps

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
                    halt_step=step if halted else None,
                )
            finally:
                if hasattr(driver, "close"):
                    driver.close()

    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
        static_config = self.config.model_copy(
            update={"n_steps": 0, "minimize": False, "thermo_freq": 1, "dump_freq": 0}
        )
        engine = LammpsEngine(static_config)
        return engine.run(structure, potential)

    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )
        with ctx:
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            script_path = temp_dir / "relax.lmp"
            with script_path.open("w") as f:
                self.generator.write_minimization_script(f, potential_path, data_file, elements)
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self._execute_simulation(driver, script_path)
                return driver.get_atoms(elements)
            finally:
                if hasattr(driver, "close"):
                    driver.close()
