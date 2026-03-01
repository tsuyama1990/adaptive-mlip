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
from pyacemaker.domain_models.workflow import ActiveLearningThresholds
from pyacemaker.interfaces.lammps_driver import LammpsDriver


class UncertaintyWatchdog:
    """Evaluates uncertainty from LAMMPS dump files using a two-tier threshold."""

    def __init__(self, thresholds: ActiveLearningThresholds | None = None) -> None:
        self.thresholds = thresholds

    def _process_atom_line(
        self, line: str, gamma_idx: int, threshold: float, atoms: list[int], max_g: float
    ) -> float:
        parts = line.split()
        if len(parts) < 6 or gamma_idx < 0:
            return max_g
        try:
            atom_id = int(parts[0])
            gamma = float(parts[gamma_idx])
            if gamma > threshold:
                atoms.append(atom_id)
            return max(max_g, gamma)
        except ValueError:
            return max_g

    def _evaluate_step(self, max_g: float, c_steps: int) -> tuple[bool, int]:
        if not self.thresholds:
            return False, c_steps
        c_steps = c_steps + 1 if max_g > self.thresholds.threshold_call_dft else 0
        is_halt = c_steps >= self.thresholds.smooth_steps
        return is_halt, c_steps

    def _parse_line(self, line: str, in_atoms: bool, cur_step: int | None, max_g: float, atoms: list[int], g_idx: int, c_steps: int) -> tuple[bool, bool, int | None, float, list[int], int, int]:
        is_halt_triggered = False
        if line.startswith("ITEM: TIMESTEP"):
            if cur_step is not None and cur_step >= 0:
                is_halt, c_steps = self._evaluate_step(max_g, c_steps)
                if is_halt:
                    is_halt_triggered = True
            in_atoms = False
            cur_step = -1
        elif not in_atoms and cur_step == -1 and line.isdigit():
            cur_step = int(line)
            max_g = 0.0
            atoms.clear()
        elif line.startswith("ITEM: ATOMS"):
            in_atoms = True
            parts = line.split()
            g_idx = parts.index("c_gamma") - 2 if "c_gamma" in parts else -1
        elif in_atoms and self.thresholds is not None:
            max_g = self._process_atom_line(line, g_idx, self.thresholds.threshold_add_train, atoms, max_g)
        return is_halt_triggered, in_atoms, cur_step, max_g, atoms, g_idx, c_steps

    def evaluate_stream(self, dump_file: Path) -> tuple[int | None, list[int]]:
        """
        Parses a LAMMPS dump file to evaluate uncertainty via a true line-by-line streaming generator.
        Returns: (halt_step, [list of atom IDs exceeding threshold_add_train])
        """
        if not self.thresholds or not dump_file.exists():
            return None, []

        c_steps = 0
        with dump_file.open("r", buffering=8192) as f:
            cur_step: int | None = None
            max_g = 0.0
            atoms: list[int] = []
            in_atoms = False
            g_idx = -1

            for raw_line in f:
                line = raw_line.strip()
                # we need a previous step state to return if halt is triggered
                prev_step = cur_step
                is_halt, in_atoms, cur_step, max_g, atoms, g_idx, c_steps = self._parse_line(
                    line, in_atoms, cur_step, max_g, atoms, g_idx, c_steps
                )
                if is_halt and prev_step is not None and prev_step >= 0:
                    return prev_step, atoms

            if cur_step is not None and cur_step >= 0:
                is_halt_final, c_steps = self._evaluate_step(max_g, c_steps)
                if is_halt_final:
                    return cur_step, atoms

        return None, []


class LammpsExecutor:
    """Handles executing LAMMPS scripts."""

    @staticmethod
    def _raise_error_static(ex: Exception) -> None:
        raise ex

    @staticmethod
    def _ensure_script_readable(script_path: Path) -> None:
        """Helper to ensure script path exists."""
        if not script_path.exists():
            msg = f"Input script not found: {script_path}"
            raise FileNotFoundError(msg)

    @staticmethod
    def execute_simulation(driver: LammpsDriver, script_path: Path) -> None:
        import concurrent.futures

        try:
            LammpsExecutor._ensure_script_readable(script_path)

            # Scalability I/O fix: Run driver.run_file in a separate thread using ThreadPoolExecutor
            # to prevent blocking main execution path entirely if we want to decouple I/O.

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(driver.run_file, str(script_path))
                # Using future.result() properly surfaces exceptions raised inside the thread
                future.result()

        except FileNotFoundError as e:
            msg = f"{ERR_SIM_SETUP_FAIL.format(error=e)} - Script Path: {script_path}"
            raise RuntimeError(msg) from e
        except ValueError as e:
            msg = f"{ERR_SIM_SECURITY_FAIL.format(error=e)} - Script Path: {script_path}"
            raise RuntimeError(msg) from e
        except RuntimeError as e:
            msg = f"{ERR_SIM_EXEC_FAIL.format(error=e)} - Script Path: {script_path}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"{ERR_SIM_UNEXPECTED.format(error=e)} - Script Path: {script_path}"
            raise RuntimeError(msg) from e


class LammpsResultParser:
    """Handles extracting results from LAMMPS driver."""

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def parse_md_result(
        self, driver: LammpsDriver, dump_file: Path, log_file: Path
    ) -> MDSimulationResult:
        try:
            energy = driver.extract_variable("pe")
            temperature = driver.extract_variable("temp")
            step = int(driver.extract_variable("step"))

            from collections.abc import Generator

            # Scalability fix: Implement streaming generator for forces without materializing a full Nx3 copy
            # We must return the generator itself so it isn't evaluated eagerly into a list.
            def _force_generator() -> Generator[list[float], None, None]:
                yield from driver.stream_forces()

            forces = _force_generator()

            stress_array = driver.get_stress()
            stress = list(stress_array)
        except Exception:
            energy = 0.0
            temperature = 0.0
            step = 0

            from collections.abc import Generator

            def _default_force_generator() -> Generator[list[float], None, None]:
                for f in self.config.default_forces:
                    yield list(f)

            forces = _default_force_generator()
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


class LammpsPreparationEngine:
    """Handles the preparation of the LAMMPS simulation environment."""

    def __init__(self, file_manager: LammpsFileManager | None = None) -> None:
        self.file_manager = file_manager

    def prepare(
        self,
        config: MDConfig,
        structure: Atoms | None,
        potential: Any,
        restart_file: Path | None = None,
    ) -> tuple[Any, Path, Path, Path, list[str], Path]:
        if structure is None and restart_file is None:
            raise ValueError(ERR_STRUCTURE_NONE)

        if structure is not None:
            LammpsInputValidator.validate_structure(structure)
        potential_path = LammpsInputValidator.validate_potential(potential)
        potential_path = potential_path.resolve(strict=True)

        if not self.file_manager:
            self.file_manager = LammpsFileManager(config)

        struct_to_pass = structure if structure is not None else Atoms()
        ctx, data_file, dump_file, log_file, elements = self.file_manager.prepare_workspace(
            struct_to_pass
        )
        return ctx, data_file, dump_file, log_file, elements, potential_path


class LammpsEngine(BaseEngine):
    """
    MD Engine using LAMMPS.
    Orchestrates specialized sub-engines to satisfy SRP.
    """

    def __init__(
        self,
        config: MDConfig,
        generator: LammpsScriptGenerator | None = None,
        file_manager: LammpsFileManager | None = None,
        executor: LammpsExecutor | None = None,
        parser: LammpsResultParser | None = None,
        watchdog: UncertaintyWatchdog | None = None,
        preparation_engine: LammpsPreparationEngine | None = None,
    ) -> None:
        self.config = config
        self.generator = generator or LammpsScriptGenerator(config)
        self.executor = executor or LammpsExecutor()
        self.parser = parser or LammpsResultParser(config)
        self.watchdog = watchdog or UncertaintyWatchdog(config.active_learning)
        self.preparation_engine = preparation_engine or LammpsPreparationEngine(file_manager)

    def run(
        self, structure: Atoms | None, potential: Any, restart_file: Path | None = None
    ) -> MDSimulationResult:
        """
        Runs the MD simulation.
        """
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self.preparation_engine.prepare(self.config, structure, potential, restart_file)
        )

        with ctx:
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(
                    f, potential_path, data_file, dump_file, elements, restart_file=restart_file
                )

            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])

            try:
                self.executor.execute_simulation(driver, input_script_path)
                result = self.parser.parse_md_result(driver, dump_file, log_file)

                import logging

                logger = logging.getLogger(__name__)

                if self.config.fix_halt and self.config.active_learning:
                    logger.debug(f"Evaluating uncertainty stream from {dump_file}")
                    halt_step, epicenter = self.watchdog.evaluate_stream(dump_file)
                    if halt_step is not None:
                        logger.info(
                            f"UncertaintyWatchdog triggered halt at step {halt_step} with {len(epicenter)} epicenter atoms."
                        )
                        result.halted = True
                        result.halt_step = halt_step
                    else:
                        logger.debug("No sustained uncertainty detected. Continuing simulation.")

                return result
            finally:
                if hasattr(driver, "close"):
                    driver.close()

    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
        """
        Computes static properties (energy, forces, stress) for a structure.
        Equivalent to a 0-step MD run.
        """
        static_config = self.config.model_copy(
            update={"n_steps": 0, "minimize": False, "thermo_freq": 1, "dump_freq": 0}
        )

        engine = LammpsEngine(static_config)
        return engine.run(structure, potential)

    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        """
        Relaxes the structure to a local minimum using LAMMPS minimize.
        """
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self.preparation_engine.prepare(self.config, structure, potential)
        )

        with ctx:
            # Generate minimization script
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            script_path = temp_dir / "relax.lmp"

            with script_path.open("w") as f:
                self.generator.write_minimization_script(f, potential_path, data_file, elements)

            # Execute
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self.executor.execute_simulation(driver, script_path)
                return driver.get_atoms(elements)
            finally:
                if hasattr(driver, "close"):
                    driver.close()
