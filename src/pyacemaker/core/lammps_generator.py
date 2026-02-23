from pathlib import Path

from ase.data import atomic_numbers

from pyacemaker.domain_models.md import MDConfig


class LammpsScriptGenerator:
    """
    Generates LAMMPS input scripts based on MDConfig.
    Follows Single Responsibility Principle by isolating script generation logic.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def _gen_potential(self, potential_path: Path, elements: list[str]) -> list[str]:
        """Generates potential definition commands."""
        lines = []
        species_str = " ".join(elements)

        if self.config.hybrid_potential:
            # Hybrid overlay: PACE + ZBL
            params = self.config.hybrid_params
            lines.append(f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}")

            # PACE
            lines.append(f"pair_coeff * * pace {potential_path} {species_str}")

            # ZBL
            n_types = len(elements)
            for i in range(n_types):
                el_i = elements[i]
                z_i = atomic_numbers[el_i]
                for j in range(i, n_types):
                    el_j = elements[j]
                    z_j = atomic_numbers[el_j]
                    lines.append(f"pair_coeff {i+1} {j+1} zbl {z_i} {z_j}")
        else:
            # Pure PACE
            lines.append("pair_style pace")
            lines.append(f"pair_coeff * * pace {potential_path} {species_str}")

        return lines

    def _gen_settings(self) -> list[str]:
        """Generates general MD settings."""
        lines = []
        lines.append(f"neighbor {self.config.neighbor_skin} bin")
        lines.append("neigh_modify delay 0 every 1 check yes")
        lines.append(f"timestep {self.config.timestep}")
        return lines

    def _gen_watchdog(self, potential_path: Path) -> list[str]:
        """Generates Uncertainty Watchdog commands."""
        lines = []
        lines.append(f"compute gamma all pace {potential_path}")
        lines.append("compute max_gamma all reduce max c_gamma")
        lines.append("variable max_g equal c_max_gamma")

        lines.append(
            f"fix halt_check all halt {self.config.check_interval} "
            f"v_max_g > {self.config.uncertainty_threshold} error continue"
        )
        return lines

    def _gen_execution(self) -> list[str]:
        """Generates minimization and MD run commands."""
        lines = []

        if self.config.minimize:
            lines.append("minimize 1.0e-4 1.0e-6 100 1000")

        # Calculate damping parameters
        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        lines.append(f"velocity all create {self.config.temperature} 12345")
        lines.append(
            f"fix npt all npt temp {self.config.temperature} {self.config.temperature} {tdamp} "
            f"iso {self.config.pressure} {self.config.pressure} {pdamp}"
        )

        lines.append(f"run {self.config.n_steps}")
        return lines

    def _gen_output(self, dump_file: Path) -> list[str]:
        """Generates output settings."""
        lines = []
        lines.append(f"thermo {self.config.thermo_freq}")
        lines.append("thermo_style custom step temp pe press v_max_g")
        lines.append(f"dump traj all custom {self.config.dump_freq} {dump_file} id type x y z c_gamma")

        # Check if halted
        lines.append("variable halted equal f_halt_check")
        lines.append("print 'Halted: ${halted}'")
        return lines

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
        lines = [
            "clear",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"read_data {data_file}",
        ]

        lines.extend(self._gen_potential(potential_path, elements))
        lines.extend(self._gen_settings())
        lines.extend(self._gen_watchdog(potential_path))
        lines.extend(self._gen_execution())
        lines.extend(self._gen_output(dump_file))

        return "\n".join(lines)
