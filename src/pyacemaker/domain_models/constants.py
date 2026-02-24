# Physics and Numerical Constants
# These are physical/mathematical constants, not configuration defaults.

import math

RECIPROCAL_FACTOR = 2 * math.pi
QE_KPOINT_TOLERANCE = 1e-3
DEFAULT_STRAIN_RANGE = 0.05
KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K
DEFAULT_RAM_DISK_PATH = "/dev/shm"  # noqa: S108
LAMMPS_SCREEN_ARG = "none"

# Security Constants
# Characters that are considered dangerous in file paths or shell commands
DANGEROUS_PATH_CHARS = [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r", "%"]

# Embedding Constants
EMBEDDING_TOLERANCE_CELL = 1e-6
EMBEDDING_TOLERANCE_LENGTH = 1e-9

# Standard LJ parameters (sigma in Angstrom, epsilon in eV)
# Used as fallback when no configuration is provided.
# Source: Generic/AMBER/UFF approximations.
# These parameters are intended for delta learning baseline (short-range repulsion).
DEFAULT_LJ_PARAMS: dict[str, dict[str, float]] = {
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
}

FALLBACK_LJ_PARAMS = {"sigma": 3.0, "epsilon": 0.01}
