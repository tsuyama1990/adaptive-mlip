from pathlib import Path
from typing import TextIO

from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import (
    DEFAULT_MD_MINIMIZE_FTOL,
    DEFAULT_MD_MINIMIZE_TOL,
    LAMMPS_MIN_STYLE_CG,
    LAMMPS_MINIMIZE_MAX_ITER,
    LAMMPS_MINIMIZE_STEPS,
    LAMMPS_VELOCITY_SEED,
)
from pyacemaker.domain_models.md import MDConfig


class LammpsScriptGenerator:
    """
    Generates LAMMPS input scripts based on MDConfig.
    Follows Single Responsibility Principle by isolating script generation logic.
    Supports writing directly to a file-like object to handle large scripts efficiently.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def _quote(self, path: Path | str) -> str:
        """Quotes a path for LAMMPS script safety."""
        return f'"{path}"'

    def _gen_potential(self, buffer: TextIO, potential_path: Path, elements: list[str]) -> None:
        """Generates potential definition commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(potential_path)

        lines = []
        if self.config.hybrid_potential:
            # Hybrid overlay: PACE + ZBL
            params = self.config.hybrid_params
            lines.append(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}\n")

            # PACE
            lines.append(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

            # ZBL
            n_types = len(elements)
            for i in range(n_types):
                el_i = elements[i]
                z_i = atomic_numbers[el_i]
                for j in range(i, n_types):
                    el_j = elements[j]
                    z_j = atomic_numbers[el_j]
                    lines.append(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}\n")
        else:
            # Pure PACE
            lines.append("pair_style pace\n")
            lines.append(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

        # Buffered write for potential lines (can be many)
        buffer.writelines(lines)

    def _gen_settings(self, buffer: TextIO) -> None:
        """Generates general MD settings."""
        lines = [
            f"neighbor {self.config.neighbor_skin} bin\n",
            "neigh_modify delay 0 every 1 check yes\n",
            f"timestep {self.config.timestep}\n",
        ]
        buffer.writelines(lines)

    def _gen_watchdog(self, buffer: TextIO, potential_path: Path) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(potential_path)
        lines = [
            f"compute gamma all pace {quoted_pot}\n",
            "compute max_gamma all reduce max c_gamma\n",
            "variable max_g equal c_max_gamma\n",
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue\n",
        ]
        buffer.writelines(lines)

    def _gen_execution(self, buffer: TextIO) -> None:
        """Generates minimization and MD run commands."""
        lines = []
        if self.config.minimize:
            lines.append(
                f"minimize {DEFAULT_MD_MINIMIZE_TOL} {DEFAULT_MD_MINIMIZE_FTOL} "
                f"{LAMMPS_MINIMIZE_STEPS} {LAMMPS_MINIMIZE_MAX_ITER}\n"
            )

        # Calculate damping parameters
        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        lines.append(f"velocity all create {self.config.temperature} {LAMMPS_VELOCITY_SEED}\n")
        lines.append(
            f"fix npt all npt temp {self.config.temperature} {self.config.temperature} {tdamp} "
            f"iso {self.config.pressure} {self.config.pressure} {pdamp}\n"
        )
        lines.append(f"run {self.config.n_steps}\n")

        buffer.writelines(lines)

    def _gen_output_setup(self, buffer: TextIO, dump_file: Path) -> None:
        """Generates output settings (thermo and dump)."""
        lines = [f"thermo {self.config.thermo_freq}\n"]

        style_parts = ["step", "temp", "pe", "press"]
        dump_parts = ["id", "type", "x", "y", "z"]

        if self.config.fix_halt:
            style_parts.append("v_max_g")
            dump_parts.append("c_gamma")

        style = " ".join(style_parts)
        dump_cols = " ".join(dump_parts)

        quoted_dump = self._quote(dump_file)
        lines.append(f"thermo_style custom {style}\n")
        lines.append(f"dump traj all custom {self.config.dump_freq} {quoted_dump} {dump_cols}\n")

        # Define variables for extraction via Python interface
        vars_to_export = ["pe", "temp", "step", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]
        for v in vars_to_export:
            lines.append(f"variable {v} equal {v}\n")

        buffer.writelines(lines)

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
        quoted_data = self._quote(data_file)

        header_lines = [
            "clear\n",
            "units metal\n",
            f"atom_style {self.config.atom_style}\n",
            "boundary p p p\n",
            f"read_data {quoted_data}\n",
        ]
        buffer.writelines(header_lines)

        self._gen_potential(buffer, potential_path, elements)
        self._gen_settings(buffer)
        self._gen_watchdog(buffer, potential_path)

        # Output setup MUST come before run
        self._gen_output_setup(buffer, dump_file)

        self._gen_execution(buffer)

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
        quoted_data = self._quote(data_file)

        header_lines = [
            "clear\n",
            "units metal\n",
            f"atom_style {self.config.atom_style}\n",
            "boundary p p p\n",
            f"read_data {quoted_data}\n",
        ]
        buffer.writelines(header_lines)

        self._gen_potential(buffer, potential_path, elements)

        settings_lines = [
            f"neighbor {self.config.neighbor_skin} bin\n",
            "neigh_modify delay 0 every 1 check yes\n",
            f"min_style {LAMMPS_MIN_STYLE_CG}\n",
            f"minimize {DEFAULT_MD_MINIMIZE_TOL} {DEFAULT_MD_MINIMIZE_FTOL} "
            f"{LAMMPS_MINIMIZE_STEPS} {LAMMPS_MINIMIZE_MAX_ITER}\n",
        ]
        buffer.writelines(settings_lines)
