from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso

from pyacemaker.constants import RECIPROCAL_FACTOR
from pyacemaker.domain_models import DFTConfig


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
        return Espresso(  # type: ignore[no-untyped-call]
            input_data=input_data,
            pseudopotentials=config.pseudopotentials,
            kpts=kpts,
        )

    def _calculate_kpoints(self, atoms: Atoms, spacing: float) -> tuple[int, int, int]:
        """
        Calculates k-point mesh based on k-spacing using vectorized operations.
        N_i = ceil(2 * pi / (|a_i| * spacing))

        Args:
            atoms: Atoms object.
            spacing: K-point spacing in 1/Angstrom.

        Returns:
            Tuple of (k_x, k_y, k_z).
        """
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        lengths = cell.lengths()
        pbc = atoms.get_pbc()  # type: ignore[no-untyped-call]

        # Use NumPy for vectorized computation
        # 1. Mask non-PBC or small dimensions (force to 1 k-point)
        # lengths < 1e-3 is effectively zero dimension
        valid_mask = pbc & (lengths >= 1e-3)

        # 2. Compute k-points for valid dimensions.
        # N is calculated as ceil( (2*pi/spacing) / L )
        # Avoid division by zero by using safe indexing or where
        # We calculate for all, then replace invalid ones with 1
        factor = RECIPROCAL_FACTOR / spacing

        # Calculate raw values where lengths > 0 to avoid warning, though valid_mask handles logic
        # Replace 0 lengths with 1.0 temporarily to avoid div/0 in pure numpy
        safe_lengths = np.where(lengths < 1e-9, 1.0, lengths)

        k_vals = np.ceil(factor / safe_lengths).astype(int)

        # Apply mask: if valid, use k_vals, else 1
        # Also ensure at least 1
        final_kpts = np.where(valid_mask, np.maximum(1, k_vals), 1)

        return tuple(final_kpts.tolist())
