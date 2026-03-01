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
        self._atomic_numbers_cache: dict[str, int] = {}

    def _get_atomic_number(self, symbol: str) -> int:
        """Cached atomic number lookup."""
        if symbol not in self._atomic_numbers_cache:
            self._atomic_numbers_cache[symbol] = atomic_numbers[symbol]
        return self._atomic_numbers_cache[symbol]

    def _quote(self, path: str) -> str:
        """
        Escapes paths for LAMMPS script safety without using shell quoting.
        """
        # Sanitize input path
        safe_path = validate_path_safe(Path(path))
        # LAMMPS requires escaping spaces with backslashes
        escaped_path = str(safe_path).replace(" ", "\\ ")
        # Do not use shlex.quote, as LAMMPS handles raw strings, not bash syntax
        return escaped_path

    def _gen_potential_pure(self, buffer: TextIO, potential_path: Path, elements: list[str]) -> None:
        """Generates pure PACE potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        buffer.write("pair_style pace\n")
        buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

    def _gen_potential_hybrid(self, buffer: TextIO, potential_path: Path, elements: list[str]) -> None:
        """Generates hybrid PACE + ZBL potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        params = self.config.hybrid_params

        buffer.write(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}\n")
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
                zbl_lines.append(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}\n")

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

    def _gen_watchdog(self, buffer: TextIO, potential_path: Path) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(str(potential_path))
        buffer.write(f"compute gamma all pace {quoted_pot}\n")
        buffer.write("compute max_gamma all reduce max c_gamma\n")
        buffer.write("variable max_g equal c_max_gamma\n")
        buffer.write(
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue\n"
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

    def _gen_minimization(self, buffer: TextIO, is_resume: bool) -> None:
        if self.config.minimize and not is_resume:
            buffer.write(
                f"minimize {self.config.minimize_tol} {self.config.minimize_ftol} "
                f"{self.config.minimize_steps} {self.config.minimize_max_iter}\n"
            )

    def _get_ramping_params(self) -> tuple[float, float, float, float]:
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
        return temp_start, temp_end, press_start, press_end

    def _gen_velocity_initialization(self, buffer: TextIO, temp_start: float, is_resume: bool) -> None:
        if not is_resume:
            buffer.write(f"velocity all create {temp_start} {self.config.velocity_seed}\n")

    def _gen_soft_start_protocol(self, buffer: TextIO, temp_start: float, temp_end: float, tdamp: float, is_resume: bool) -> None:
        if is_resume:
            buffer.write(f"fix soft_start all langevin {temp_start} {temp_end} {tdamp / 10.0} {self.config.velocity_seed}\n")
            buffer.write("fix soft_nve all nve\n")
            buffer.write("run 100\n")
            buffer.write("unfix soft_start\n")
            buffer.write("unfix soft_nve\n")

    def _gen_npt_ensemble(self, buffer: TextIO, temp_start: float, temp_end: float, press_start: float, press_end: float, tdamp: float, pdamp: float) -> None:
        buffer.write(
            f"fix npt all npt temp {temp_start} {temp_end} {tdamp} "
            f"iso {press_start} {press_end} {pdamp}\n"
        )
        buffer.write(f"run {self.config.n_steps}\n")

    def _gen_execution(self, buffer: TextIO, elements: list[str], is_resume: bool = False) -> None:
        """Generates minimization and MD run commands by composing specific stages."""
        self._gen_minimization(buffer, is_resume)
        self._gen_mc(buffer, elements)

        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        temp_s, temp_e, press_s, press_e = self._get_ramping_params()

        self._gen_velocity_initialization(buffer, temp_s, is_resume)
        self._gen_soft_start_protocol(buffer, temp_s, temp_e, tdamp, is_resume)
        self._gen_npt_ensemble(buffer, temp_s, temp_e, press_s, press_e, tdamp, pdamp)

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
        restart_file: Path | None = None
    ) -> None:
        """
        Writes the LAMMPS input script to the provided buffer.
        """
        quoted_data = self._quote(str(data_file))

        buffer.write("clear\n")
        buffer.write("units metal\n")

        if restart_file:
             quoted_restart = self._quote(str(restart_file))
             buffer.write(f"read_restart {quoted_restart}\n")
        else:
             # Use .value to ensure we get the string value "atomic", "charge" etc.
             buffer.write(f"atom_style {self.config.atom_style.value}\n")
             buffer.write("boundary p p p\n")
             buffer.write(f"read_data {quoted_data}\n")

        self._gen_potential(buffer, potential_path, elements)
        self._gen_settings(buffer)
        self._gen_watchdog(buffer, potential_path)

        # Configure write_restart before run to save states
        quoted_restart_out = self._quote(str(dump_file.with_suffix(".restart")))
        buffer.write(f"restart 1000 {quoted_restart_out}\n")

        # Output setup MUST come before run
        self._gen_output_setup(buffer, dump_file)

        self._gen_execution(buffer, elements, is_resume=(restart_file is not None))

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
