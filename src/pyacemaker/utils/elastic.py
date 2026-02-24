from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import GPa

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.validation import (
    ElasticConfig,
    ElasticResult,
    ValidationStatus,
)
from pyacemaker.utils.lammps import run_static_lammps


class ElasticCalculator:
    """Calculates elastic constants (Cij) and checks mechanical stability."""

    def __init__(self, config: ElasticConfig, md_config: MDConfig) -> None:
        self.config = config
        self.md_config = md_config

    def _apply_strain(self, atoms: Atoms, strain_vector: np.ndarray) -> Atoms:
        """
        Applies strain to the atoms object.
        strain_vector: [e1, e2, e3, e4, e5, e6] (Voigt)
        e1=exx, e2=eyy, e3=ezz, e4=2eyz, e5=2exz, e6=2exy
        """
        strained = atoms.copy()  # type: ignore[no-untyped-call]
        cell = strained.get_cell()

        # Construct strain matrix from Voigt vector
        # e4 = 2*eyz => eyz = e4/2
        exx, eyy, ezz = strain_vector[0], strain_vector[1], strain_vector[2]
        eyz, exz, exy = strain_vector[3] / 2.0, strain_vector[4] / 2.0, strain_vector[5] / 2.0

        strain_matrix = np.array([[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]])

        # Deformation matrix F = I + epsilon (for small strains)
        deformation = np.eye(3) + strain_matrix

        # Apply deformation to cell
        new_cell = np.dot(cell, deformation)
        strained.set_cell(new_cell, scale_atoms=True)
        return strained  # type: ignore[no-any-return]

    def calculate(self, structure: Atoms, potential_path: Path) -> ElasticResult:
        """
        Calculates Cij matrix and checks Born stability.
        """
        delta = self.config.strain_magnitude
        C = np.zeros((6, 6))

        # We will calculate each column of C by applying strain e_j
        # We use central difference: e_j = +delta, -delta

        for j in range(6):
            strain_vec_plus = np.zeros(6)
            strain_vec_plus[j] = delta
            # Note: if j >= 3 (shear), the actual strain tensor component is delta/2.
            # But Voigt strain is delta.
            # So apply_strain handles dividing by 2.

            strain_vec_minus = np.zeros(6)
            strain_vec_minus[j] = -delta

            # Plus
            atoms_plus = self._apply_strain(structure, strain_vec_plus)
            _, _, stress_plus = run_static_lammps(atoms_plus, potential_path, self.md_config)

            # Minus
            atoms_minus = self._apply_strain(structure, strain_vec_minus)
            _, _, stress_minus = run_static_lammps(atoms_minus, potential_path, self.md_config)

            # Calculate slope: sigma_i / e_j
            # stress is [sxx, syy, szz, syz, sxz, sxy] matching Voigt 1..6
            diff_stress = (stress_plus - stress_minus) / (2 * delta)
            C[:, j] = diff_stress

        # Symmetrize C (optional but physically required)
        C = (C + C.T) / 2.0

        # Check stability: Eigenvalues of C must be positive
        eigenvalues = np.linalg.eigvalsh(C)
        is_stable = np.all(eigenvalues > 0)

        # Calculate Bulk Modulus (Voigt average)
        B_v = ((C[0, 0] + C[1, 1] + C[2, 2]) + 2 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9.0

        # Convert to GPa for report
        c_ij_gpa = C / GPa
        bulk_modulus_gpa = B_v / GPa

        # Format Cij dict
        c_ij_dict = {}
        for i in range(6):
            for j in range(i, 6):  # Upper triangle
                c_ij_dict[f"C{i + 1}{j + 1}"] = float(c_ij_gpa[i, j])

        status = ValidationStatus.PASS if is_stable else ValidationStatus.FAIL

        return ElasticResult(
            c_ij=c_ij_dict,
            bulk_modulus=float(bulk_modulus_gpa),
            is_mechanically_stable=bool(is_stable),
            status=status,
        )
