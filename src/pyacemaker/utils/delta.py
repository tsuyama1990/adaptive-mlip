
import numpy as np
from ase.data import atomic_numbers

# Standard LJ parameters (sigma in Angstrom, epsilon in eV)
# These are rough estimates or standard values from literature (e.g., UFF, AMBER)
# For the purpose of delta learning baseline, they just need to be repulsive at short range.
LJ_PARAMS = {
    "H": {"sigma": 2.57, "epsilon": 0.004},
    "He": {"sigma": 2.10, "epsilon": 0.001},
    "Li": {"sigma": 2.18, "epsilon": 0.01},
    "Be": {"sigma": 2.45, "epsilon": 0.05},
    "B": {"sigma": 3.64, "epsilon": 0.004},
    "C": {"sigma": 3.40, "epsilon": 0.005},
    "N": {"sigma": 3.25, "epsilon": 0.007},
    "O": {"sigma": 3.02, "epsilon": 0.009},
    "F": {"sigma": 2.95, "epsilon": 0.011},
    "Ne": {"sigma": 2.72, "epsilon": 0.013},
    "Na": {"sigma": 2.66, "epsilon": 0.015},
    "Mg": {"sigma": 2.69, "epsilon": 0.017},
    "Al": {"sigma": 4.01, "epsilon": 0.021},
    "Si": {"sigma": 3.83, "epsilon": 0.017},
    "P": {"sigma": 3.69, "epsilon": 0.013},
    "S": {"sigma": 3.59, "epsilon": 0.011},
    "Cl": {"sigma": 3.51, "epsilon": 0.010},
    "Ar": {"sigma": 3.40, "epsilon": 0.010},
    "K": {"sigma": 3.40, "epsilon": 0.010},
    "Ca": {"sigma": 3.03, "epsilon": 0.012},
    "Sc": {"sigma": 2.94, "epsilon": 0.013},
    "Ti": {"sigma": 2.83, "epsilon": 0.014},
    "V": {"sigma": 2.73, "epsilon": 0.016},
    "Cr": {"sigma": 2.69, "epsilon": 0.016},
    "Mn": {"sigma": 2.64, "epsilon": 0.015},
    "Fe": {"sigma": 2.59, "epsilon": 0.013},
    "Co": {"sigma": 2.56, "epsilon": 0.012},
    "Ni": {"sigma": 2.52, "epsilon": 0.011},
    "Cu": {"sigma": 3.11, "epsilon": 0.005},
    "Zn": {"sigma": 2.46, "epsilon": 0.012},
    # Add more as needed or implement a scaling rule based on atomic radius
}

DEFAULT_LJ_PARAMS = {"sigma": 3.0, "epsilon": 0.01}


def get_lj_params(element: str) -> dict[str, float]:
    """
    Returns Lennard-Jones parameters for a given element.
    If the element is not found in the database, returns generic default parameters.

    Args:
        element: Chemical symbol (e.g., "Fe", "O").

    Returns:
        Dictionary with "sigma" (Angstrom) and "epsilon" (eV).
    """
    return LJ_PARAMS.get(element, DEFAULT_LJ_PARAMS.copy())


def compute_zbl_energy(el1: str, el2: str, r: float) -> float:
    """
    Computes the ZBL potential energy for a pair of atoms.

    Args:
        el1: Chemical symbol of first atom.
        el2: Chemical symbol of second atom.
        r: Distance between atoms in Angstrom.

    Returns:
        Energy in eV.
    """
    if r <= 0:
        msg = "Distance must be positive."
        raise ValueError(msg)

    z1 = atomic_numbers[el1]
    z2 = atomic_numbers[el2]

    # Constants
    e_squared = 14.3996  # eV * Angstrom (Coulomb constant)
    a0 = 0.529177  # Bohr radius in Angstrom

    # Screening length
    a = (0.8854 * a0) / (z1**0.23 + z2**0.23)

    x = r / a

    # Universal screening function
    phi = (
        0.1818 * np.exp(-3.2 * x)
        + 0.5099 * np.exp(-0.9423 * x)
        + 0.2802 * np.exp(-0.4029 * x)
        + 0.02817 * np.exp(-0.2016 * x)
    )

    return float((z1 * z2 * e_squared / r) * phi)
