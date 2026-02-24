from collections.abc import Callable
from typing import Any, cast

import numpy as np
from ase import Atoms


class ElasticCalculator:
    """
    Calculates Elastic Constants (Cij) and Moduli using finite differences.
    """

    def __init__(self, atoms: Atoms, strain: float = 0.01) -> None:
        self.atoms = atoms
        self.strain = strain

    def calculate(self, stress_fn: Callable[[Atoms], Any]) -> tuple[list[list[float]], float, float]:
        """
        Calculates Cij matrix and moduli.

        Args:
            stress_fn: Function taking Atoms and returning stress (Voigt form 6-vector or 3x3 matrix).
                       ASE get_stress() typically returns Voigt 6-vector [xx, yy, zz, yz, xz, xy].

        Returns:
            Cij (6x6 list), Bulk_Modulus (GPa), Shear_Modulus (GPa)
        """
        # Voigt notation indices
        # 0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy

        C = np.zeros((6, 6))

        # We need to apply independent strains.
        for i in range(6):
            # Apply +strain
            atoms_plus = self._apply_strain(self.atoms, i, self.strain)
            stress_plus = self._ensure_voigt(stress_fn(atoms_plus))

            # Apply -strain
            atoms_minus = self._apply_strain(self.atoms, i, -self.strain)
            stress_minus = self._ensure_voigt(stress_fn(atoms_minus))

            # Central difference
            delta_stress = stress_plus - stress_minus
            C[:, i] = delta_stress / (2 * self.strain)

        # Convert to GPa (assuming stress was in eV/A^3)
        # 1 eV/A^3 = 160.21766208 GPa
        GPA_CONVERSION = 160.21766208
        C *= GPA_CONVERSION

        # Symmetrize
        C = (C + C.T) / 2.0

        # Calculate Voigt averages for B and G
        B_V = (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[1,2] + C[0,2])) / 9.0
        G_V = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[1,2]+C[0,2]) + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0

        return C.tolist(), float(B_V), float(G_V)

    def _apply_strain(self, atoms: Atoms, index: int, value: float) -> Atoms:
        """Applies Voigt strain to atoms."""
        strain_tensor = np.zeros((3, 3))
        # Map Voigt index to tensor
        if index == 0:
            strain_tensor[0, 0] = value
        elif index == 1:
            strain_tensor[1, 1] = value
        elif index == 2:
            strain_tensor[2, 2] = value
        elif index == 3:
            strain_tensor[1, 2] = value / 2.0
            strain_tensor[2, 1] = value / 2.0
        elif index == 4:
            strain_tensor[0, 2] = value / 2.0
            strain_tensor[2, 0] = value / 2.0
        elif index == 5:
            strain_tensor[0, 1] = value / 2.0
            strain_tensor[1, 0] = value / 2.0

        deformation = np.eye(3) + strain_tensor

        new_atoms = cast(Atoms, atoms.copy())  # type: ignore[no-untyped-call]
        new_cell = np.dot(atoms.get_cell(), deformation.T)  # type: ignore[no-untyped-call] # Check transposes
        new_atoms.set_cell(new_cell, scale_atoms=True)  # type: ignore[no-untyped-call]
        return new_atoms

    def _ensure_voigt(self, stress: Any) -> np.ndarray:
        """Ensures stress is a 6-element numpy array."""
        s = np.array(stress)
        if s.shape == (3, 3):
            # Convert 3x3 to Voigt [xx, yy, zz, yz, xz, xy]
            return np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])
        if s.shape == (6,):
            return s
        msg = f"Invalid stress shape: {s.shape}"
        raise ValueError(msg)

    # For mocking/testing internal methods if needed
    def _compute_stress_tensor(self, atoms: Atoms, calc_fn: Callable[..., Any]) -> np.ndarray:
        return self._ensure_voigt(calc_fn(atoms))

    def _fit_elastic_constants(self, strains: Any, stresses: Any) -> tuple[list[float], float, float]:
        # Placeholder if we used fitting, but we used finite diff above
        return [], 0.0, 0.0

    @staticmethod
    def check_stability(bulk_modulus: float, shear_modulus: float) -> bool:
        """
        Basic mechanical stability check (Born criteria for isotropic).
        """
        return bulk_modulus > 0 and shear_modulus > 0
