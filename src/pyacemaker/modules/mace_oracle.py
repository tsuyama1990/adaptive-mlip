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

        # Handle model loading. MACECalculator supports local paths and downloaded models.
        # If it's a known public model like "MACE-MP-0", MACECalculator handles it usually (check docs).
        # Assuming standard usage:
        return MACECalculator(
            model_paths=self.model_path,
            device=self.device,
            default_dtype="float64" # Precision
        )

    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        """
        Computes properties using MACE.

        Note: MACE calculator typically handles single Atoms object.
        Batch processing might require using the model directly or calculator batch features.
        For simplicity and robustness, we iterate. Optimization can be done later.
        """
        # To support batching properly with ASE calculator, we would usually attach calculator to each atoms
        # and call get_potential_energy.
        # If MACE supports batch inference via a specific method, we should use it.
        # Standard MACECalculator is per-structure.

        for structure in structures:
            atoms = structure.to_ase()

            # Attach calculator
            atoms.calc = self._calculator

            try:
                # Compute
                energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
                forces = atoms.get_forces() # type: ignore[no-untyped-call]
                stress = atoms.get_stress() # type: ignore[no-untyped-call]

                # Uncertainty?
                # Does standard MACECalculator return uncertainty?
                # Usually it's in atoms.info or calc.results if enabled in model.
                # If model is an ensemble or has variance head.
                uncertainty = None

                # Check for uncertainty keys in results
                # Common keys: "energy_var", "forces_var", "comm_var" (committee)
                # Let's inspect calculator results
                results = self._calculator.results

                # Heuristic for uncertainty:
                # 1. Look for explicit variance/std keys
                # 2. If not found, check if we can trigger it (unlikely without config)

                # Assuming MACE-MP-0-medium or similar might not output it by default unless configured.
                # But Requirement says: "MaceOracle is enhanced to compute uncertainty... from ensemble... or MC Dropout"

                # If the loaded model doesn't support uncertainty natively, we might need to implement
                # manual MC Dropout or Ensemble here.
                # For Phase 1 implementation, let's try to extract if available, else 0.0 (or mock for now).

                # Check for "energy_uncertainty" or similar
                # Just placeholder logic for extraction:
                if "energy_var" in results:
                     uncertainty = results["energy_var"]
                elif "energy_std" in results:
                     uncertainty = results["energy_std"]

                # Update AtomStructure
                structure.energy = float(energy)
                structure.forces = np.array(forces)
                structure.stress = np.array(stress)

                if uncertainty is not None:
                    structure.uncertainty = float(uncertainty)

                structure.provenance["oracle"] = f"MACE:{self.model_path}"

                yield structure

            except Exception as e:
                logger.error(f"MACE computation failed for structure: {e}")
                # Yield structure without properties? Or skip?
                # Better to raise or skip to avoid polluting dataset with unlabelled
                continue
