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
FILENAME_TRAINING: Final = "training.pckl"
FILENAME_POTENTIAL: Final = "potential.yace"

# LAMMPS
LAMMPS_SCREEN_ARG: Final = "none"
LAMMPS_SAFE_CMD_PATTERN: Final = r"^[a-zA-Z0-9_\-\.\/\s\*\+\=\<\>\!\@\#\$\%\^\&\(\)\{\}\[\]\:\;\'\"]*$"

# Defaults
DEFAULT_RAM_DISK_PATH: Final = "/dev/shm"
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
