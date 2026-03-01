import shlex
from pathlib import Path
from typing import TextIO

from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import (
    LAMMPS_CMD_BOUNDARY_P,
    LAMMPS_CMD_CLEAR,
    LAMMPS_CMD_COMPUTE_GAMMA,
    LAMMPS_CMD_COMPUTE_MAX_GAMMA,
    LAMMPS_CMD_DUMP,
    LAMMPS_CMD_FIX_HALT,
    LAMMPS_CMD_FIX_NPT,
    LAMMPS_CMD_MINIMIZE,
    LAMMPS_CMD_NEIGHBOR_DELAY,
    LAMMPS_CMD_RUN,
    LAMMPS_CMD_THERMO,
    LAMMPS_CMD_THERMO_STYLE,
    LAMMPS_CMD_UNITS_METAL,
    LAMMPS_CMD_VAR_MAX_G,
    LAMMPS_CMD_VELOCITY_CREATE,
    LAMMPS_MIN_STYLE_CG,
    LAMMPS_PAIR_STYLE_HYBRID_PACE_ZBL,
    LAMMPS_PAIR_STYLE_PACE,
    MAX_PATH_LENGTH,
)
from pyacemaker.domain_models.md import HybridParams, MDConfig
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
        self._quote_cache: dict[str, str] = {}

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
        if len(path) > MAX_PATH_LENGTH:
            msg = "Path too long"
            raise ValueError(msg)

        if path not in self._quote_cache:
            # Sanitize input path
            safe_path = validate_path_safe(Path(path))
            # Use shlex.quote for shell safety
            quoted = shlex.quote(str(safe_path))
            # Validate the quoted path doesn't introduce vulnerabilities
            validate_path_safe(Path(quoted.strip("'\"")))
            self._quote_cache[path] = quoted
        return self._quote_cache[path]

    def _gen_potential_pure(
        self, buffer: TextIO, potential_path: Path, elements: list[str]
    ) -> None:
        """Generates pure PACE potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        buffer.write(f"{LAMMPS_PAIR_STYLE_PACE}\n")
        buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

    def _gen_potential_hybrid(
        self, buffer: TextIO, potential_path: Path, elements: list[str], params: HybridParams
    ) -> None:
        """Generates hybrid PACE + ZBL potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))

        pair_style = LAMMPS_PAIR_STYLE_HYBRID_PACE_ZBL.format(
            inner=params.zbl_cut_inner, outer=params.zbl_cut_outer
        )
        buffer.write(f"{pair_style}\n")
        buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

        import itertools

        n_types = len(elements)

        # Optimization: Use generator expression for O(1) memory overhead and direct writes
        buffer.writelines(
            f"pair_coeff {i + 1} {j + 1} zbl {self._get_atomic_number(elements[i])} {self._get_atomic_number(elements[j])}\n"
            for i, j in itertools.combinations_with_replacement(range(n_types), 2)
        )

    def _gen_potential(self, buffer: TextIO, potential_path: Path, elements: list[str]) -> None:
        """Generates potential definition commands."""
        if self.config.hybrid_potential:
            self._gen_potential_hybrid(buffer, potential_path, elements, self.config.hybrid_params)
        else:
            self._gen_potential_pure(buffer, potential_path, elements)

    def _gen_settings(self, buffer: TextIO) -> None:
        """Generates general MD settings."""
        buffer.write(f"neighbor {self.config.neighbor_skin} bin\n")
        buffer.write(LAMMPS_CMD_NEIGHBOR_DELAY)
        buffer.write(f"timestep {self.config.timestep}\n")

    def _gen_watchdog(self, buffer: TextIO, potential_path: Path) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(str(potential_path))
        buffer.write(LAMMPS_CMD_COMPUTE_GAMMA.format(pot=quoted_pot))
        buffer.write(LAMMPS_CMD_COMPUTE_MAX_GAMMA)
        buffer.write(LAMMPS_CMD_VAR_MAX_G)
        buffer.write(
            LAMMPS_CMD_FIX_HALT.format(
                interval=self.config.check_interval,
                threshold=self.config.uncertainty_threshold
            )
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
                LAMMPS_CMD_MINIMIZE.format(
                    tol=self.config.minimize_tol,
                    ftol=self.config.minimize_ftol,
                    steps=self.config.minimize_steps,
                    max_iter=self.config.minimize_max_iter
                )
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
        buffer.write(LAMMPS_CMD_VELOCITY_CREATE.format(temp=temp_start, seed=self.config.velocity_seed))
        buffer.write(
            LAMMPS_CMD_FIX_NPT.format(
                t_start=temp_start, t_end=temp_end, tdamp=tdamp,
                p_start=press_start, p_end=press_end, pdamp=pdamp
            )
        )
        buffer.write(LAMMPS_CMD_RUN.format(steps=self.config.n_steps))

    def _gen_output_setup(self, buffer: TextIO, dump_file: Path) -> None:
        """Generates output settings (thermo and dump)."""
        buffer.write(LAMMPS_CMD_THERMO.format(freq=self.config.thermo_freq))

        style_parts = ["step", "temp", "pe", "press"]
        dump_parts = ["id", "type", "x", "y", "z"]

        if self.config.fix_halt:
            style_parts.append("v_max_g")
            dump_parts.append("c_gamma")

        style = " ".join(style_parts)
        dump_cols = " ".join(dump_parts)

        quoted_dump = self._quote(str(dump_file))
        buffer.write(LAMMPS_CMD_THERMO_STYLE.format(style=style))
        buffer.write(LAMMPS_CMD_DUMP.format(freq=self.config.dump_freq, dump=quoted_dump, cols=dump_cols))

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

        buffer.write(LAMMPS_CMD_CLEAR)
        buffer.write(LAMMPS_CMD_UNITS_METAL)
        # Use .value to ensure we get the string value "atomic", "charge" etc.
        buffer.write(f"atom_style {self.config.atom_style.value}\n")
        buffer.write(LAMMPS_CMD_BOUNDARY_P)
        buffer.write(f"read_data {quoted_data}\n")

        self._gen_potential(buffer, potential_path, elements)
        self._gen_settings(buffer)
        self._gen_watchdog(buffer, potential_path)

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

        buffer.write(LAMMPS_CMD_CLEAR)
        buffer.write(LAMMPS_CMD_UNITS_METAL)
        buffer.write(f"atom_style {self.config.atom_style.value}\n")
        buffer.write(LAMMPS_CMD_BOUNDARY_P)
        buffer.write(f"read_data {quoted_data}\n")

        self._gen_potential(buffer, potential_path, elements)

        buffer.write(f"neighbor {self.config.neighbor_skin} bin\n")
        buffer.write(LAMMPS_CMD_NEIGHBOR_DELAY)
        buffer.write(f"min_style {LAMMPS_MIN_STYLE_CG}\n")
        buffer.write(
            LAMMPS_CMD_MINIMIZE.format(
                tol=self.config.minimize_tol,
                ftol=self.config.minimize_ftol,
                steps=self.config.minimize_steps,
                max_iter=self.config.minimize_max_iter
            )
        )
