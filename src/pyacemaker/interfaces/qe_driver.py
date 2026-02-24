from functools import lru_cache
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.data import chemical_symbols

from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import RECIPROCAL_FACTOR


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

        Raises:
            ValueError: If configuration parameters are invalid/unsafe.
        """
        # Security: Validate sensitive parameters (though Pydantic does most heavy lifting)
        # Here we double check constraints relevant to runtime context
        if config.encut <= 0:
            msg = "Energy cutoff must be positive."
            raise ValueError(msg)
        if config.kpoints_density <= 0:
            msg = "K-points density must be positive."
            raise ValueError(msg)

        # Validate pseudopotential keys
        valid_symbols = set(chemical_symbols)
        for elem in config.pseudopotentials:
            if elem not in valid_symbols:
                msg = f"Invalid chemical symbol in pseudopotentials: {elem}"
                raise ValueError(msg)

        # Calculate k-points
        # For memoization, we need immutable inputs. Atoms is mutable.
        # We pass full cell matrix (tuple of tuples) and PBC to handle general cells correctly.
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        cell_tuple = tuple(tuple(float(x) for x in row) for row in cell)
        pbc = tuple(atoms.get_pbc())  # type: ignore[no-untyped-call]

        kpts = self._calculate_kpoints_cached(cell_tuple, pbc, config.kpoints_density)

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
        return Espresso(  # type: ignore[no-untyped-call]
            input_data=input_data,
            pseudopotentials=config.pseudopotentials,
            kpts=kpts,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_kpoints_cached(
        cell_tuple: tuple[tuple[float, ...], ...], pbc: tuple[bool, ...], spacing: float
    ) -> tuple[int, int, int]:
        """
        Calculates k-point mesh based on k-spacing for general cells.
        Cached version for performance.

        Args:
            cell_tuple: Cell matrix as tuple of tuples (row vectors).
            pbc: PBC flags.
            spacing: K-point spacing in reciprocal Angstrom.

        Returns:
            Tuple of (k_x, k_y, k_z).
        """
        cell = np.array(cell_tuple)

        # Handle degenerate/zero cells safely
        # Check volume
        try:
            # 3x3 determinant
            vol = np.abs(np.linalg.det(cell))
        except np.linalg.LinAlgError:
            vol = 0.0

        if vol < 1e-9:
            return (1, 1, 1)

        # Reciprocal lattice vectors b_i (without 2pi factor yet for simplicity in cross products)

        a1, a2, a3 = cell[0], cell[1], cell[2]

        cross_lengths = [
            np.linalg.norm(np.cross(a2, a3)),  # for b1
            np.linalg.norm(np.cross(a3, a1)),  # for b2
            np.linalg.norm(np.cross(a1, a2)),  # for b3
        ]

        kpts = []
        for i in range(3):
            if not pbc[i]:
                kpts.append(1)
                continue

            # |b_i| = 2*pi * cross_len / V
            b_norm = (RECIPROCAL_FACTOR * cross_lengths[i]) / vol

            # N_i = ceil( |b_i| / spacing )
            # Avoid division by zero if spacing is tiny (but config validator handles <=0)
            val = int(np.ceil(b_norm / spacing))
            kpts.append(max(1, val))

        return tuple(kpts)  # type: ignore[return-value]
