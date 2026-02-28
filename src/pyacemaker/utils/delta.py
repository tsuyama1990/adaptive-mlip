import numpy as np
from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import DEFAULT_LJ_PARAMS, FALLBACK_LJ_PARAMS


def get_lj_params(element: str) -> dict[str, float]:
    """
    Returns Lennard-Jones parameters for a given element.
    If the element is not found in the database, returns generic default parameters.

    Args:
        element: Chemical symbol (e.g., "Fe", "O").

    Returns:
        Dictionary with "sigma" (Angstrom) and "epsilon" (eV).
    """
    return DEFAULT_LJ_PARAMS.get(element, FALLBACK_LJ_PARAMS.copy()) # type: ignore[return-value]


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

    # Screening length (ZBL Universal Screening)
    # 0.8854 is the screening length constant for the Thomas-Fermi model approximation
    a = (0.8854 * a0) / (z1**0.23 + z2**0.23)

    x = r / a

    # Universal screening function (Ziegler-Biersack-Littmark)
    # The coefficients are empirically fitted to interatomic potentials
    phi = (
        0.1818 * np.exp(-3.2 * x)
        + 0.5099 * np.exp(-0.9423 * x)
        + 0.2802 * np.exp(-0.4029 * x)
        + 0.02817 * np.exp(-0.2016 * x)
    )

    return float((z1 * z2 * e_squared / r) * phi)
