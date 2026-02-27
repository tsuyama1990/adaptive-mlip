from collections.abc import Iterator
from typing import Any

import numpy as np

from pyacemaker.core.base import BaseOracle
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.logger import get_logger

logger = get_logger()

class MaceOracle(BaseOracle):
    """
    Oracle wrapping a MACE model for energy, forces, and uncertainty prediction.
    """
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """
        Args:
            model_path: Path to the MACE model file (.pt or .model) or model name (e.g. MACE-MP-0).
            device: Calculation device ('cpu', 'cuda').
        """
        self.model_path = model_path
        self.device = device
        self._calculator = self._load_calculator()

    def _load_calculator(self) -> Any:
        """Loads the MACE calculator."""
        try:
            # Check availability
            from mace.calculators import MACECalculator
        except ImportError:
            raise ImportError("mace-torch is not installed. Please install it to use MaceOracle.")

        logger.info(f"Loading MACE model from {self.model_path} on {self.device}...")

        # Robust loading
        try:
            return MACECalculator(
                model_paths=self.model_path,
                device=self.device,
                default_dtype="float64"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MACE model: {e}") from e

    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        """
        Computes properties using MACE.

        Raises:
            RuntimeError: If computation fails for a batch and cannot be recovered.
        """
        for structure in structures:
            atoms = structure.to_ase()

            # Attach calculator
            atoms.calc = self._calculator

            try:
                # Compute
                # This triggers calculation
                energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
                forces = atoms.get_forces() # type: ignore[no-untyped-call]
                stress = atoms.get_stress() # type: ignore[no-untyped-call]

                # Extract Uncertainty
                uncertainty = None
                results = self._calculator.results

                # Standard keys for uncertainty in various MACE versions/configs
                # If not present, uncertainty remains None (handled by Active Learning step)
                if "energy_var" in results:
                     uncertainty = results["energy_var"]
                elif "energy_std" in results:
                     uncertainty = results["energy_std"]
                elif "forces_var" in results:
                     # Aggregate force variance if energy variance is missing?
                     # Heuristic: max force variance
                     uncertainty = np.max(results["forces_var"])

                # Update AtomStructure
                structure.energy = float(energy)
                structure.forces = np.array(forces)
                structure.stress = np.array(stress)

                if uncertainty is not None:
                    structure.uncertainty = float(uncertainty)

                structure.provenance["oracle"] = f"MACE:{self.model_path}"

                yield structure

            except Exception as e:
                # Error handling strategy:
                # Log error and skip structure? Or fail pipeline?
                # Active learning usually tolerates some failures.
                # However, silent failure is bad.
                # We log strictly.
                logger.error(f"MACE computation failed for structure: {e}")
                # We yield the structure WITHOUT updating properties implies it failed.
                # But AtomStructure with None energy is invalid for training?
                # Better to NOT yield it if we can't label it.
                # Thus filtering out failed calculations.
                continue
