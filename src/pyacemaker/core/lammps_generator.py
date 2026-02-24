from io import StringIO
from pathlib import Path

from ase.data import atomic_numbers

from pyacemaker.domain_models.md import MDConfig


class LammpsScriptGenerator:
    """
    Generates LAMMPS input scripts based on MDConfig.
    Follows Single Responsibility Principle by isolating script generation logic.
    Uses StringIO for efficient string construction.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def _quote(self, path: Path | str) -> str:
        """Quotes a path for LAMMPS script safety."""
        return f'"{path}"'

    def _gen_potential(self, buffer: StringIO, potential_path: Path, elements: list[str]) -> None:
        """Generates potential definition commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(potential_path)

        if self.config.hybrid_potential:
            # Hybrid overlay: PACE + ZBL
            params = self.config.hybrid_params
            buffer.write(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}\n")

            # PACE
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

    def _gen_settings(self, buffer: StringIO) -> None:
        """Generates general MD settings."""
        buffer.write(f"neighbor {self.config.neighbor_skin} bin\n")
        buffer.write("neigh_modify delay 0 every 1 check yes\n")
        buffer.write(f"timestep {self.config.timestep}\n")

    def _gen_watchdog(self, buffer: StringIO, potential_path: Path) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(potential_path)
        buffer.write(f"compute gamma all pace {quoted_pot}\n")
        buffer.write("compute max_gamma all reduce max c_gamma\n")
        buffer.write("variable max_g equal c_max_gamma\n")

        buffer.write(
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue\n"
        )

    def _gen_execution(self, buffer: StringIO) -> None:
        """Generates minimization and MD run commands."""
        if self.config.minimize:
            buffer.write("minimize 1.0e-4 1.0e-6 100 1000\n")

        # Calculate damping parameters
        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        buffer.write(f"velocity all create {self.config.temperature} 12345\n")
        buffer.write(
            f"fix npt all npt temp {self.config.temperature} {self.config.temperature} {tdamp} "
            f"iso {self.config.pressure} {self.config.pressure} {pdamp}\n"
        )

        buffer.write(f"run {self.config.n_steps}\n")

    def _gen_output_setup(self, buffer: StringIO, dump_file: Path) -> None:
        """Generates output settings (thermo and dump)."""
        buffer.write(f"thermo {self.config.thermo_freq}\n")

        style = "step temp pe press"
        dump_cols = "id type x y z"

        if self.config.fix_halt:
            style += " v_max_g"
            dump_cols += " c_gamma"

        quoted_dump = self._quote(dump_file)
        buffer.write(f"thermo_style custom {style}\n")
        buffer.write(f"dump traj all custom {self.config.dump_freq} {quoted_dump} {dump_cols}\n")

    def _gen_post_run_diagnostics(self, buffer: StringIO) -> None:
        """Generates post-run diagnostic prints."""
        # Check if halted logic was intended here, but strictly relying on step count in Python is safer.
        # Keeping this method for future extensibility or if explicit status print is needed.

    def generate(
        self,
        potential_path: Path,
        data_file: Path,
        dump_file: Path,
        elements: list[str],
    ) -> str:
        """
        Orchestrates LAMMPS input script generation.
        """
        buffer = StringIO()

        quoted_data = self._quote(data_file)

        buffer.write("clear\n")
        buffer.write("units metal\n")
        buffer.write(f"atom_style {self.config.atom_style}\n")
        buffer.write("boundary p p p\n")
        buffer.write(f"read_data {quoted_data}\n")

        self._gen_potential(buffer, potential_path, elements)
        self._gen_settings(buffer)
        self._gen_watchdog(buffer, potential_path)

        # Output setup MUST come before run
        self._gen_output_setup(buffer, dump_file)

        self._gen_execution(buffer)

        self._gen_post_run_diagnostics(buffer)

        return buffer.getvalue()
