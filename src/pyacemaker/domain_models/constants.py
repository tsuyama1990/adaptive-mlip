from typing import Final

import numpy as np

# Physics
RECIPROCAL_FACTOR: Final = 2 * np.pi

# Tolerances
EMBEDDING_TOLERANCE_CELL: Final = 1e-3

# Lennard-Jones
DEFAULT_LJ_PARAMS: Final = {
    "Fe": {"sigma": 2.3, "epsilon": 0.5},
    "O": {"sigma": 3.0, "epsilon": 0.01},
}
FALLBACK_LJ_PARAMS: Final = {"sigma": 2.5, "epsilon": 0.1}

# Security
DANGEROUS_PATH_CHARS: Final = frozenset([";", "&", "|", "`", "$", "(", ")", "<", ">"])

# File names
FILENAME_CANDIDATES: Final = "candidates.xyz"
# Changed from .pckl to .xyz to support robust streaming append
FILENAME_TRAINING: Final = "training.xyz"
FILENAME_POTENTIAL: Final = "potential.yace"

# LAMMPS
LAMMPS_SCREEN_ARG: Final = "none"
# Removed & and ; and | from safe pattern to prevent command injection
LAMMPS_SAFE_CMD_PATTERN: Final = r"^[a-zA-Z0-9_\-\.\/\s\*\+\=\<\>\!\@\#\$\%\^\(\)\{\}\[\]\:\'\"]*$"

# Defaults
DEFAULT_RAM_DISK_PATH: Final = "/dev/shm"  # noqa: S108
DEFAULT_N_CANDIDATES: Final = 100
DEFAULT_BATCH_SIZE: Final = 50
DEFAULT_CHECKPOINT_INTERVAL: Final = 10
DEFAULT_DATA_DIR: Final = "data"
DEFAULT_ACTIVE_LEARNING_DIR: Final = "active_learning"
DEFAULT_POTENTIALS_DIR: Final = "potentials"
DEFAULT_STATE_FILE: Final = "state.json"
DEFAULT_PRODUCTION_DIR: Final = "production"

# Validation
DEFAULT_STRAIN_RANGE: Final = 0.05

# Test
TEST_ENERGY_H2O: Final = -14.2
TEST_ENERGY_GENERIC: Final = -5.0

# LAMMPS Commands
LMP_CMD_CLEAR: Final = "clear"
LMP_CMD_UNITS: Final = "units metal"
LMP_CMD_BOUNDARY: Final = "boundary p p p"
LMP_CMD_NEIGHBOR: Final = "neighbor {skin} bin"
LMP_CMD_NEIGH_MODIFY: Final = "neigh_modify delay 0 every 1 check yes"
LMP_CMD_MIN_STYLE: Final = "min_style cg"
LMP_CMD_MINIMIZE: Final = "minimize 1.0e-6 1.0e-8 1000 10000"
LMP_CMD_READ_DATA: Final = 'read_data "{data_file}"'
LMP_CMD_ATOM_STYLE: Final = "atom_style {style}"

LAMMPS_VELOCITY_SEED: Final = 12345
LAMMPS_MINIMIZE_STEPS: Final = 100
LAMMPS_MINIMIZE_MAX_ITER: Final = 1000
LAMMPS_MIN_STYLE_CG: Final = "cg"
DEFAULT_MD_MINIMIZE_TOL: Final = 1.0e-4
DEFAULT_MD_MINIMIZE_FTOL: Final = 1.0e-6

# Error Messages
ERR_SIM_SETUP_FAIL: Final = "Simulation setup failed: {error}"
ERR_SIM_SECURITY_FAIL: Final = "Simulation security validation failed: {error}"
ERR_SIM_EXEC_FAIL: Final = "LAMMPS engine execution failed: {error}"
ERR_SIM_UNEXPECTED: Final = "Unexpected error during simulation execution: {error}"
ERR_POTENTIAL_NOT_FOUND: Final = "Potential file not found: {path}"
ERR_STRUCTURE_NONE: Final = "Structure cannot be None after validation."

# Validation
ERR_VAL_STRUCT_NONE: Final = "Structure must be provided."
ERR_VAL_STRUCT_TYPE: Final = "Expected ASE Atoms object, got {type}."
ERR_VAL_STRUCT_EMPTY: Final = "Structure contains no atoms."
ERR_VAL_POT_NONE: Final = "Potential path must be provided."
ERR_VAL_POT_NOT_FILE: Final = "Potential path is not a file: {path}"
ERR_VAL_POT_OUTSIDE: Final = "Potential path {path} is outside allowed directories (CWD, /tmp, /dev/shm)."

# Oracle
ERR_ORACLE_ITERATOR: Final = "Input 'structures' must be an Iterator (got {type}). Use iter() to create one."
ERR_ORACLE_FAILED: Final = "DFT calculation failed after {attempts} attempts."

# M3GNet
ERR_M3GNET_PRED_FAIL: Final = "M3GNet prediction failed for {composition}"

# Generator
ERR_GEN_NCAND_NEG: Final = "n_candidates must be non-negative, got {n}"
ERR_GEN_BASE_FAIL: Final = "Failed to generate base structure for {composition}: {error}"

# Validator
ERR_VAL_REQ_STRUCT: Final = "Validation requires a structure."
