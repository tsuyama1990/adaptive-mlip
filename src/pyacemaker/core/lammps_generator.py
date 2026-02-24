from pathlib import Path
from typing import TextIO

from ase.data import atomic_numbers

from pyacemaker.domain_models.defaults import (
    DEFAULT_MD_ATOM_STYLE,
    DEFAULT_MD_MINIMIZE_FTOL,
    DEFAULT_MD_MINIMIZE_TOL,
)
from pyacemaker.domain_models.md import MDConfig


class LammpsScriptGenerator:
    """
    Generates LAMMPS input scripts based on MDConfig.
    Follows Single Responsibility Principle by isolating script generation logic.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def _quote(self, path: Path | str) -> str:
        """Quotes a path for LAMMPS script safety."""
        # Performance: Use simple string concatenation
        return '"' + str(path) + '"'

    def _gen_potential(self, buffer: TextIO, potential_path: Path, elements: list[str]) -> None:
        """Generates potential definition commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(potential_path)

        if self.config.hybrid_potential:
            # Hybrid overlay: PACE + ZBL
            params = self.config.hybrid_params

            # Use configurable cutoffs from HybridParams
            outer = params.zbl_global_cutoff
            inner = outer * 0.8  # Heuristic if not specified

            # Using f-strings is generally efficient in Python 3.12+
            buffer.write(f"pair_style hybrid/overlay pace zbl {inner} {outer}\n")
            buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

            # ZBL
            n_types = len(elements)
            for i in range(n_types):
                el_i = elements[i]
                z_i = atomic_numbers[el_i]
                for j in range(i, n_types):
                    el_j = elements[j]
                    z_j = atomic_numbers[el_j]
                    buffer.write(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}\n")
        else:
            # Pure PACE
            buffer.write("pair_style pace\n")
            buffer.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

    def _gen_settings(self, buffer: TextIO) -> None:
        """Generates general MD settings."""
        # Batch write for performance
        lines = [
            f"neighbor {self.config.neighbor_skin} bin",
            "neigh_modify delay 0 every 1 check yes",
            f"timestep {self.config.timestep}",
        ]
        buffer.write("\n".join(lines) + "\n")

    def _gen_watchdog(self, buffer: TextIO, potential_path: Path) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(potential_path)
        lines = [
            f"compute gamma all pace {quoted_pot}",
            "compute max_gamma all reduce max c_gamma",
            "variable max_g equal c_max_gamma",
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue",
        ]
        buffer.write("\n".join(lines) + "\n")

    def _gen_execution(self, buffer: TextIO) -> None:
        """Generates minimization and MD run commands."""
        if self.config.minimize:
            buffer.write(f"minimize {DEFAULT_MD_MINIMIZE_TOL} {DEFAULT_MD_MINIMIZE_FTOL} 100 1000\n")

        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        buffer.write(f"velocity all create {self.config.temperature} 12345\n")
        buffer.write(
            f"fix npt all npt temp {self.config.temperature} {self.config.temperature} {tdamp} "
            f"iso {self.config.pressure} {self.config.pressure} {pdamp}\n"
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

        quoted_dump = self._quote(dump_file)

        lines = [
            f"thermo_style custom {style}",
            f"dump traj all custom {self.config.dump_freq} {quoted_dump} {dump_cols}",
            # Variables for Python extraction
            "variable pe equal pe",
            "variable temp equal temp",
            "variable step equal step",
            "variable pxx equal pxx",
            "variable pyy equal pyy",
            "variable pzz equal pzz",
            "variable pxy equal pxy",
            "variable pxz equal pxz",
            "variable pyz equal pyz",
        ]
        buffer.write("\n".join(lines) + "\n")

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

        header = [
            "clear",
            "units metal",
            f"atom_style {DEFAULT_MD_ATOM_STYLE}",
            "boundary p p p",
            f"read_data {quoted_data}",
        ]
        buffer.write("\n".join(header) + "\n")

        self._gen_potential(buffer, potential_path, elements)
        self._gen_settings(buffer)
        self._gen_watchdog(buffer, potential_path)
        self._gen_output_setup(buffer, dump_file)
        self._gen_execution(buffer)

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

        lines = [
            "clear",
            "units metal",
            f"atom_style {DEFAULT_MD_ATOM_STYLE}",
            "boundary p p p",
            f"read_data {quoted_data}",
        ]
        buffer.write("\n".join(lines) + "\n")

        self._gen_potential(buffer, potential_path, elements)

        buffer.write(f"neighbor {self.config.neighbor_skin} bin\n")
        buffer.write("neigh_modify delay 0 every 1 check yes\n")

        buffer.write("min_style cg\n")
        buffer.write(f"minimize {DEFAULT_MD_MINIMIZE_TOL} {DEFAULT_MD_MINIMIZE_FTOL} 100 10000\n")
