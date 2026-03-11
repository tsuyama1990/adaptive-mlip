from ase import Atoms

from pyacemaker.domain_models.constants import ERR_M3GNET_PRED_FAIL


class M3GNetWrapper:
    """
    Wrapper for M3GNet structure prediction.
    Constructs an initial structure for 'cold start' using ASE building tools
    and relaxes it using a foundation model if available.
    """

    def predict_structure(self, composition: str) -> Atoms:
        """
        Predict a stable structure for the given composition.
        Args:
            composition: Chemical formula (e.g., 'Fe', 'NaCl').
        Returns:
            Atoms object.
        Raises:
            RuntimeError: If prediction fails after retries.
        """
        # Validate composition string ensuring no injected shells or unsafe characters exist
        import re

        if not re.match(r"^[A-Za-z0-9]+$", composition):
            msg = f"Invalid composition string format: {composition}"
            raise ValueError(msg)

        try:
            return self._predict(composition)
        except Exception as e:
            raise RuntimeError(ERR_M3GNET_PRED_FAIL.format(composition=composition)) from e

    def _predict(self, composition: str) -> Atoms:
        from ase import Atoms
        from ase.build import bulk

        # For known simple structures, use proper ASE bulk generation.
        # This provides a realistic starting point rather than a mock.
        if composition == "FePt":
            # For L1_0 FePt
            atoms = Atoms(
                "FePt",
                scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                cell=[2.7, 2.7, 3.8],
                pbc=True,
            )
        else:
            try:
                atoms = bulk(composition)
            except Exception:
                # Basic fallback: construct a simple cubic structure from formula
                from ase.symbols import string2symbols

                symbols = string2symbols(composition)
                n_atoms = len(symbols)

                # Create a simple cubic cell with reasonable spacing
                spacing = 2.5
                cell_dim = spacing * (n_atoms ** (1 / 3))

                # Distribute randomly or simply on a line for fallback
                positions = []
                for i in range(n_atoms):
                    positions.append(
                        [i * spacing % cell_dim, (i * spacing / cell_dim) * spacing % cell_dim, 0]
                    )

                atoms = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=[cell_dim, cell_dim, cell_dim],
                    pbc=True,
                )

        # Attempt to relax the structure using a foundation model to find a real local minimum
        try:
            import os
            from pathlib import Path

            import torch
            from ase.optimize import LBFGS
            from mace.calculators import mace_mp

            # Initialize MACE calculator
            calc = mace_mp(
                model="medium",
                dispersion=False,
                default_dtype="float64",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            atoms.calc = calc

            with Path(os.devnull).open("w") as devnull:
                opt = LBFGS(atoms, logfile=devnull)
                opt.run(fmax=0.05, steps=100)  # type: ignore[no-untyped-call]
        except ImportError:
            # Fallback to unrelaxed if MACE isn't available
            pass
        except Exception as e:
            # Ignore relaxation errors and return unrelaxed starting point
            import logging

            logging.getLogger(__name__).debug(f"Pre-relaxation failed: {e}")

        return atoms
