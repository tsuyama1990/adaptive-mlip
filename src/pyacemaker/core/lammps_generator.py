import shlex
from pathlib import Path
from typing import TextIO

from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import LAMMPS_MIN_STYLE_CG
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.utils.path import validate_path_safe


class LammpsScriptGenerator:
    """
    Generates LAMMPS input scripts based on MDConfig.
    Follows Single Responsibility Principle by isolating script generation logic.
    Supports writing directly to a file-like object to handle large scripts efficiently.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config
        # Use lru_cache for methods instead of manual dict
        self._atomic_numbers_cache: dict[str, int] = {}

    def _get_atomic_number(self, symbol: str) -> int:
        """Cached atomic number lookup."""
        if symbol not in self._atomic_numbers_cache:
            self._atomic_numbers_cache[symbol] = atomic_numbers[symbol]
        return self._atomic_numbers_cache[symbol]

    def _quote(self, path: str) -> str:
        """
        Quotes a path for LAMMPS script safety after validation.
        Uses caching to avoid redundant validation calls.
        """
        # Sanitize input path
        # Note: path must be string for lru_cache
        safe_path = validate_path_safe(Path(path))
        if not safe_path.exists() and not safe_path.parent.exists():
            msg = f"Path {safe_path} is invalid or has an invalid parent directory."
            raise ValueError(msg)

        # Use shlex.quote for shell safety
        return shlex.quote(str(safe_path))

    def _gen_potential_pure(
        self, buffer: TextIO, potential_path: Path, elements: list[str]
    ) -> None:
        """Generates pure PACE potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        buffer.write("pair_style pace\n")
        buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

    def _gen_potential_hybrid(
        self, buffer: TextIO, potential_path: Path, elements: list[str]
    ) -> None:
        """Generates hybrid PACE + ZBL potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        params = self.config.hybrid_params

        buffer.write(
            f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}\n"
        )
        buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

        n_types = len(elements)

        # Optimize loop string concatenation
        # Use list comprehension for ZBL pairs
        zbl_lines = []
        for i in range(n_types):
            el_i = elements[i]
            z_i = self._get_atomic_number(el_i)
            for j in range(i, n_types):
                el_j = elements[j]
                z_j = self._get_atomic_number(el_j)
                zbl_lines.append(f"pair_coeff {i + 1} {j + 1} zbl {z_i} {z_j}\n")

        buffer.writelines(zbl_lines)

    def _gen_potential(self, buffer: TextIO, potential_path: Path, elements: list[str]) -> None:
        """Generates potential definition commands."""
        if self.config.hybrid_potential:
            self._gen_potential_hybrid(buffer, potential_path, elements)
        else:
            self._gen_potential_pure(buffer, potential_path, elements)

    def _gen_settings(self, buffer: TextIO) -> None:
        """Generates general MD settings."""
        buffer.write(f"neighbor {self.config.neighbor_skin} bin\n")
        buffer.write("neigh_modify delay 0 every 1 check yes\n")
        buffer.write(f"timestep {self.config.timestep}\n")

    def _gen_watchdog(self, buffer: TextIO, potential_path: Path, data_file: Path | None = None) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(str(potential_path))
        buffer.write(f"compute gamma all pace {quoted_pot}\n")
        buffer.write("compute max_gamma all reduce max c_gamma\n")
        buffer.write("variable max_g equal c_max_gamma\n")

        smooth_steps = getattr(self.config, 'smooth_steps', 3)
        threshold_call_dft = self.config.uncertainty_threshold

        # Secure file-based Python execution (no inline code)
        evaluator_path = "evaluator.py"
        if data_file:
            evaluator_path_obj = data_file.parent / "evaluator.py"
            evaluator_path = self._quote(str(evaluator_path_obj))

            # Write the static evaluator script directly to the workspace
            with evaluator_path_obj.open("w") as f:
                f.write("def invoke_evaluator():\n")
                f.write("    from lammps import lammps\n")
                f.write("    lmp = lammps(ptr=lammps.get_lammps_ptr())\n")
                f.write("    try:\n")
                f.write("        max_g = float(lmp.extract_variable('max_g', 0, 0))\n")
                f.write("        threshold = float(lmp.extract_variable('threshold_call_dft', 0, 0))\n")
                f.write("        smooth_steps = int(lmp.extract_variable('smooth_steps', 0, 0))\n")
                f.write("    except Exception as e:\n")
                f.write("        print(f'Evaluator variable extraction failed: {e}')\n")
                f.write("        return\n")
                f.write("    \n")
                f.write("    # Use LAMMPS internal variable for state instead of python sys.globals\n")
                f.write("    try:\n")
                f.write("        exceed_count = int(lmp.extract_variable('exceed_count', 0, 0))\n")
                f.write("    except Exception:\n")
                f.write("        exceed_count = 0\n")
                f.write("    \n")
                f.write("    if max_g > threshold:\n")
                f.write("        exceed_count += 1\n")
                f.write("        lmp.command(f'variable exceed_count equal {exceed_count}')\n")
                f.write("        if exceed_count >= smooth_steps:\n")
                f.write("            print('True Anomaly Detected, Halting')\n")
                f.write("            lmp.command('quit')\n")
                f.write("        else:\n")
                f.write("            print('Thermal Noise Detected, Ignoring')\n")
                f.write("    else:\n")
                f.write("        lmp.command('variable exceed_count equal 0')\n")

        buffer.write(f"variable threshold_call_dft equal {threshold_call_dft}\n")
        buffer.write(f"variable smooth_steps equal {smooth_steps}\n")
        buffer.write("variable exceed_count equal 0\n")

        buffer.write(f"python invoke_evaluator invoke {evaluator_path}\n")
        buffer.write(
            f"fix python_invoke all python/invoke {self.config.check_interval} "
            f"post_force invoke_evaluator\n"
        )
    def _gen_mc(self, buffer: TextIO, elements: list[str]) -> None:
        """Generates Monte Carlo atom swapping commands."""
        if not self.config.mc:
            return

        n_types = len(elements)
        if n_types < 2:
            return  # Can't swap if fewer than 2 types

        # types keyword requires list of types to swap
        types_str = " ".join(str(i + 1) for i in range(n_types))

        # Command syntax: fix mc all atom/swap N X seed T types {types}
        # N: swap frequency (steps)
        # X: swaps per attempt (set to 1)
        # T: temperature (for Boltzmann factor)

        temp = self.config.temperature
        if self.config.ramping and self.config.ramping.temp_start is not None:
            temp = self.config.ramping.temp_start

        buffer.write(
            f"fix mc_swap all atom/swap {self.config.mc.swap_freq} 1 {self.config.mc.seed} "
            f"{temp} ke no types {types_str}\n"
        )

    def _gen_execution(self, buffer: TextIO, elements: list[str]) -> None:
        """Generates minimization and MD run commands."""
        if self.config.minimize:
            buffer.write(
                f"minimize {self.config.minimize_tol} {self.config.minimize_ftol} "
                f"{self.config.minimize_steps} {self.config.minimize_max_iter}\n"
            )

        # MC
        self._gen_mc(buffer, elements)

        # Calculate damping parameters
        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        # Determine T/P start/end
        temp_start = self.config.temperature
        temp_end = self.config.temperature
        press_start = self.config.pressure
        press_end = self.config.pressure

        if self.config.ramping:
            if self.config.ramping.temp_start is not None:
                temp_start = self.config.ramping.temp_start
            if self.config.ramping.temp_end is not None:
                temp_end = self.config.ramping.temp_end
            if self.config.ramping.press_start is not None:
                press_start = self.config.ramping.press_start
            if self.config.ramping.press_end is not None:
                press_end = self.config.ramping.press_end

        # Use configurable velocity seed
        buffer.write(f"velocity all create {temp_start} {self.config.velocity_seed}\n")
        buffer.write(
            f"fix npt all npt temp {temp_start} {temp_end} {tdamp} "
            f"iso {press_start} {press_end} {pdamp}\n"
        )
        buffer.write(f"run {self.config.n_steps}\n")

    def _gen_output_setup(self, buffer: TextIO, dump_file: Path) -> None:
        """Generates output settings (thermo and dump)."""
        buffer.write(f"thermo {self.config.thermo_freq}\n")

        style_parts = ["step", "temp", "pe", "press"]
        dump_parts = ["id", "type", "x", "y", "z"]

        if self.config.fix_halt:
            style_parts.append("v_max_g")
            dump_parts.append("c_gamma")

        style = " ".join(style_parts)
        dump_cols = " ".join(dump_parts)

        quoted_dump = self._quote(str(dump_file))
        buffer.write(f"thermo_style custom {style}\n")
        buffer.write(f"dump traj all custom {self.config.dump_freq} {quoted_dump} {dump_cols}\n")

        # Define variables for extraction via Python interface
        vars_to_export = ["pe", "temp", "step", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]
        for v in vars_to_export:
            buffer.write(f"variable {v} equal {v}\n")

    def _gen_post_run_diagnostics(self, buffer: TextIO) -> None:
        """Generates post-run diagnostic prints."""

    def write_script(
        self,
        buffer: TextIO,
        potential_path: Path,
        data_file: Path,
        dump_file: Path,
        elements: list[str],
    ) -> None:
        """
        Writes the LAMMPS input script to the provided buffer.
        """
        quoted_data = self._quote(str(data_file))

        buffer.write("clear\n")
        buffer.write("units metal\n")
        # Use .value to ensure we get the string value "atomic", "charge" etc.
        buffer.write(f"atom_style {self.config.atom_style.value}\n")
        buffer.write("boundary p p p\n")
        buffer.write(f"read_data {quoted_data}\n")

        self._gen_potential(buffer, potential_path, elements)
        self._gen_settings(buffer)
        self._gen_watchdog(buffer, potential_path, data_file)

        # Output setup MUST come before run
        self._gen_output_setup(buffer, dump_file)

        self._gen_execution(buffer, elements)

        self._gen_post_run_diagnostics(buffer)

    def write_minimization_script(
        self,
        buffer: TextIO,
        potential_path: Path,
        data_file: Path,
        elements: list[str],
    ) -> None:
        """
        Writes a minimization-only script for relaxation.
        """
        quoted_data = self._quote(str(data_file))

        buffer.write("clear\n")
        buffer.write("units metal\n")
        buffer.write(f"atom_style {self.config.atom_style.value}\n")
        buffer.write("boundary p p p\n")
        buffer.write(f"read_data {quoted_data}\n")

        self._gen_potential(buffer, potential_path, elements)

        buffer.write(f"neighbor {self.config.neighbor_skin} bin\n")
        buffer.write("neigh_modify delay 0 every 1 check yes\n")
        buffer.write(f"min_style {LAMMPS_MIN_STYLE_CG}\n")
        buffer.write(
            f"minimize {self.config.minimize_tol} {self.config.minimize_ftol} "
            f"{self.config.minimize_steps} {self.config.minimize_max_iter}\n"
        )
