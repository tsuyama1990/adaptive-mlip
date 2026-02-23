import math
from typing import Any

from ase import Atoms
from ase.calculators.espresso import Espresso

from pyacemaker.domain_models import DFTConfig

# Physics constant for Reciprocal Lattice Vector conversion
# b_i = 2 * pi / a_i (for orthogonal cells)
RECIPROCAL_FACTOR = 2 * math.pi


class QEDriver:
    """
    Interface for Quantum Espresso calculations using ASE.
    """

    def get_calculator(self, atoms: Atoms, config: DFTConfig) -> Espresso:
        """
        Creates an ASE Espresso calculator configured based on DFTConfig.

        Args:
            atoms: The atoms object to calculate (used for k-point generation).
            config: The DFT configuration.

        Returns:
            Configured Espresso calculator.
        """
        # Calculate k-points
        kpts = self._calculate_kpoints(atoms, config.kpoints_density)

        # Construct input data
        input_data: dict[str, Any] = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "disk_io": "low",  # Optimize I/O
                # pseudo_dir and outdir are typically handled by ASE or env vars
            },
            "system": {
                "ecutwfc": config.encut,
                "occupations": "smearing",
                "smearing": config.smearing_type,
                "degauss": config.smearing_width,
            },
            "electrons": {
                "mixing_beta": config.mixing_beta,
                "diagonalization": config.diagonalization,
                "conv_thr": 1.0e-8,
            },
        }

        # Create calculator
        return Espresso(
            input_data=input_data,
            pseudopotentials=config.pseudopotentials,
            kpts=kpts,
        )

    def _calculate_kpoints(self, atoms: Atoms, spacing: float) -> tuple[int, int, int]:
        """
        Calculates k-point mesh based on k-spacing.
        N_i = ceil(2 * pi / (|a_i| * spacing))

        Args:
            atoms: Atoms object.
            spacing: K-point spacing in 1/Angstrom.

        Returns:
            Tuple of (k_x, k_y, k_z).
        """
        cell = atoms.get_cell()
        # lengths() call is cheap (returns cached lengths if cell hasn't changed, or simple norm)
        lengths = cell.lengths()
        pbc = atoms.get_pbc()

        # Optimize loop: direct computation
        kpts_list: list[int] = []

        # Precompute constant
        factor = RECIPROCAL_FACTOR / spacing

        for i in range(3):
            if not pbc[i] or lengths[i] < 1e-3:
                kpts_list.append(1)
            else:
                # N = (2*pi/L) / spacing = (2*pi/spacing) / L
                k_val = math.ceil(factor / lengths[i])
                kpts_list.append(max(1, int(k_val)))

        return tuple(kpts_list) # type: ignore
