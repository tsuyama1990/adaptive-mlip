from functools import lru_cache
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.data import chemical_symbols

from pyacemaker.constants import QE_KPOINT_TOLERANCE, RECIPROCAL_FACTOR
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

        # Validate pseudopotential keys
        valid_symbols = set(chemical_symbols)
        for elem in config.pseudopotentials:
            if elem not in valid_symbols:
                msg = f"Invalid chemical symbol in pseudopotentials: {elem}"
                raise ValueError(msg)

        # Calculate k-points
        # For memoization, we need immutable inputs. Atoms is mutable.
        # We pass cell parameters (lengths) and PBC to a helper method.
        # But cell lengths alone isn't enough for non-orthogonal, though we assumed orthogonal
        # in the logic. Let's stick to lengths+pbc for now as per current implementation.
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        cell_lengths = tuple(cell.lengths())
        pbc = tuple(atoms.get_pbc())  # type: ignore[no-untyped-call]

        # Audit Fix: Caching is done on immutable types (tuples), avoiding mutable Atoms references.
        kpts = self._calculate_kpoints_cached(cell_lengths, pbc, config.kpoints_density)

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
        lengths: tuple[float, ...], pbc: tuple[bool, ...], spacing: float
    ) -> tuple[int, int, int]:
        """
        Calculates k-point mesh based on k-spacing using vectorized operations.
        Cached version for performance (No N+1 queries).

        Args:
            lengths: Cell lengths (tuple for immutability/hashing).
            pbc: PBC flags (tuple for immutability/hashing).
            spacing: K-point spacing in 1/Angstrom.

        Returns:
            Tuple of (k_x, k_y, k_z).
        """
        # Convert inputs to numpy for vectorization
        lengths_arr = np.array(lengths)
        pbc_arr = np.array(pbc)

        # Use NumPy for vectorized computation
        # 1. Mask non-PBC or small dimensions (force to 1 k-point)
        # lengths < QE_KPOINT_TOLERANCE is effectively zero dimension
        valid_mask = pbc_arr & (lengths_arr >= QE_KPOINT_TOLERANCE)

        # 2. Compute k-points for valid dimensions.
        # N is calculated as ceil( (2*pi/spacing) / L )
        # Avoid division by zero by using safe indexing or where
        # We calculate for all, then replace invalid ones with 1
        factor = RECIPROCAL_FACTOR / spacing

        # Calculate raw values where lengths > 0 to avoid warning, though valid_mask handles logic
        # Replace 0 lengths with 1.0 temporarily to avoid div/0 in pure numpy
        safe_lengths = np.where(lengths_arr < 1e-9, 1.0, lengths_arr)

        # Use np.ceil which is efficient. Standard numpy vectorization avoids loops.
        # Float to int conversion is negligible compared to the loop cost it replaces.
        k_vals = np.ceil(factor / safe_lengths).astype(int)

        # Apply mask: if valid, use k_vals, else 1
        # Also ensure at least 1
        final_kpts = np.where(valid_mask, np.maximum(1, k_vals), 1)

        # The result must be exactly Tuple[int, int, int] but numpy returns NDArray
        # We rely on final_kpts having shape (3,)
        return tuple(final_kpts.tolist())
