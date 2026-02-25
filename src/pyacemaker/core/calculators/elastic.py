import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.domain_models.defaults import (
    DEFAULT_VALIDATION_ELASTIC_STEPS,
    DEFAULT_VALIDATION_ELASTIC_STRAIN,
)


class ElasticCalculator:
    """
    Calculates elastic constants (C_ij) and checks Born stability criteria.
    Currently supports Cubic symmetry (C11, C12, C44).
    """

    def __init__(
        self,
        engine: BaseEngine,
        strain: float = DEFAULT_VALIDATION_ELASTIC_STRAIN,
        steps: int = DEFAULT_VALIDATION_ELASTIC_STEPS,
    ) -> None:
        self.engine = engine
        self.strain = strain
        self.steps = steps

    def _get_stress(self, atoms: Atoms, potential_path: Path) -> np.ndarray:
        """Helper to get stress from engine (Voigt: xx, yy, zz, yz, xz, xy)."""
        result = self.engine.compute_static_properties(atoms, potential_path)
        return np.array(result.stress)

    def calculate_properties(
        self, structure: Atoms, potential_path: Path
    ) -> tuple[bool, dict[str, float], float, str]:
        """
        Calculates C_ij and B, checks stability.
        Returns (is_stable, c_ij, bulk_modulus, plot_base64).
        """
        # 1. Strain Sweep for C11, C12
        # Apply strain e_xx from -strain to +strain
        strains = np.linspace(-self.strain, self.strain, self.steps)
        stress_xx = []
        stress_yy = []

        base_cell = structure.get_cell()  # type: ignore[no-untyped-call]

        # Use a single structure object and modify/restore cell to avoid copying overhead
        atoms = structure.copy()  # type: ignore[no-untyped-call]
        for eps in strains:
            cell = base_cell.copy()
            cell[0, 0] *= (1 + eps)
            atoms.set_cell(cell, scale_atoms=True)

            stress = self._get_stress(atoms, potential_path)
            stress_xx.append(stress[0])
            stress_yy.append(stress[1])

        # Restore structure
        atoms.set_cell(base_cell, scale_atoms=True)

        # Fit
        c11 = np.polyfit(strains, stress_xx, 1)[0]
        c12 = np.polyfit(strains, stress_yy, 1)[0]

        # 2. Strain Sweep for C44
        # Apply shear e_xy (engineering strain gamma_xy = 2*e_xy)
        # Voigt notation: sigma_xy = C44 * gamma_xy
        stress_xy = []
        # Reuse atoms object
        atoms.set_cell(base_cell, scale_atoms=True)
        for eps in strains:
            cell = base_cell.copy()
            # Shear in xy
            # cell[0, 1] += eps * cell[1, 1] ? No, simpler for cubic
            # A simple shear matrix: [[1, eps, 0], [0, 1, 0], [0, 0, 1]] applied to cell
            # But ASE set_cell with scale_atoms=True works on basis vectors.
            # Shear strain gamma: x -> x + gamma*y
            # cell vector a1 = (a, 0, 0), a2 = (0, a, 0)
            # New a2 = (gamma*a, a, 0)
            cell[1, 0] += eps * base_cell[0, 0] # Simple shear
            atoms.set_cell(cell, scale_atoms=True)

            stress = self._get_stress(atoms, potential_path)
            stress_xy.append(stress[5]) # xy component (Voigt index 5)

        # Fit sigma_xy vs gamma (gamma = eps if defined as such)
        # Actually standard definition: e_xy = 0.5 * gamma.
        # But Voigt stress-strain relation uses gamma.
        # If I applied strain such that new_x = x + eps*y, then gamma = eps.
        c44 = np.polyfit(strains, stress_xy, 1)[0]

        # Convert units if needed (Bar -> GPa).
        # LAMMPS 'metal' units: pressure in bars.
        # 1 Bar = 0.0001 GPa.
        # 1 GPa = 10000 Bar.
        # If _get_stress returns Bar, we multiply by 1e-4.
        # But wait, fitting slope of Stress(Bar) vs Strain(unitless) gives C_ij in Bar.
        # Convert to GPa: C_GPa = C_Bar * 1e-4.

        # But _get_stress in LammpsDriver returns what?
        # extract_variable returns float. pxx is in pressure units (bars for metal).
        # So I will convert.
        CONV = 1e-4
        c11 *= CONV
        c12 *= CONV
        c44 *= CONV

        c_ij = {"C11": float(c11), "C12": float(c12), "C44": float(c44)}

        # Bulk Modulus (Cubic)
        B = (c11 + 2 * c12) / 3.0

        # Stability
        is_stable = self.check_stability_criteria(c_ij, "cubic")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(strains, stress_xx, 'o-', label='sigma_xx')
        ax1.plot(strains, stress_yy, 's-', label='sigma_yy')
        ax1.set_xlabel('Strain')
        ax1.set_ylabel('Stress (Bar)')
        ax1.legend()
        ax1.set_title('Normal Strain')

        ax2.plot(strains, stress_xy, '^-', label='sigma_xy')
        ax2.set_xlabel('Shear Strain')
        ax2.set_ylabel('Stress (Bar)')
        ax2.legend()
        ax2.set_title('Shear Strain')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return is_stable, c_ij, float(B), plot_base64

    @staticmethod
    def check_stability_criteria(c_ij: dict[str, float], system: str = "cubic") -> bool:
        if system == "cubic":
            c11 = c_ij.get("C11", 0.0)
            c12 = c_ij.get("C12", 0.0)
            c44 = c_ij.get("C44", 0.0)
            return (c11 - c12 > 0) and (c11 + 2 * c12 > 0) and (c44 > 0)
        return False

    @staticmethod
    def calculate_bulk_modulus(c_ij: dict[str, float], system: str = "cubic") -> float:
        if system == "cubic":
            c11 = c_ij.get("C11", 0.0)
            c12 = c_ij.get("C12", 0.0)
            return (c11 + 2 * c12) / 3.0
        return 0.0
