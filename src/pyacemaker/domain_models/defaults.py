import tempfile
from pathlib import Path
from typing import Final

# Configuration Defaults
DEFAULT_STATE_FILE = "state.json"
DEFAULT_DATA_DIR = "data"
DEFAULT_ACTIVE_LEARNING_DIR = "active_learning"
DEFAULT_POTENTIALS_DIR = "potentials"
DEFAULT_PRODUCTION_DIR = "production"
DEFAULT_BATCH_SIZE = 5
DEFAULT_N_CANDIDATES = 10
DEFAULT_CHECKPOINT_INTERVAL = 1

# EON Defaults
DEFAULT_EON_EXECUTABLE = "eonclient"
# Use environment variable for seed if available, otherwise None (to let random module decide or user override)
DEFAULT_EON_SEED = 12345

# File names
FILENAME_CANDIDATES = "candidates.xyz"
FILENAME_TRAINING = "training_data.xyz"
FILENAME_POTENTIAL = "potential.yace"

# Template strings
TEMPLATE_ITER_DIR = "iter_{iteration:03d}"
TEMPLATE_POTENTIAL_FILE = "generation_{iteration:03d}.yace"

# Logging Messages
LOG_PROJECT_INIT = "Project: {project_name} initialized."
LOG_CONFIG_LOADED = "Configuration loaded successfully."
LOG_DRY_RUN_COMPLETE = "Dry run complete. Configuration is valid."
LOG_START_LOOP = "Starting active learning loop."
LOG_START_ITERATION = "Starting iteration {iteration}/{max_iterations}."
LOG_ITERATION_COMPLETED = "Iteration {iteration} completed."
LOG_WORKFLOW_COMPLETED = "Active learning workflow completed successfully."
LOG_WORKFLOW_CRASHED = "Workflow crashed: {error}"
LOG_INIT_MODULES = "Initializing modules..."
LOG_MODULES_INIT_SUCCESS = "Modules initialized successfully."
LOG_MODULE_INIT_FAIL = "Module initialization failed: {error}"
LOG_GENERATED_CANDIDATES = "Generated {count} candidate structures."
LOG_COMPUTED_PROPERTIES = "Computed properties for {count} structures."
LOG_POTENTIAL_TRAINED = "Potential trained successfully."
LOG_MD_COMPLETED = "MD simulation completed."
LOG_STATE_SAVED = "State saved: {state}"
LOG_STATE_SAVE_FAIL = "Failed to save state: {error}"
LOG_STATE_LOAD_SUCCESS = "Loaded state from file. Resuming from iteration {iteration}."
LOG_STATE_LOAD_FAIL = "Failed to load state: {error}. Starting from scratch."

# Error Messages
ERR_CONFIG_NOT_FOUND = "Configuration file not found: {path}"
ERR_PATH_NOT_FILE = "Path is not a file: {path}"
ERR_PATH_TRAVERSAL = "Path traversal detected: {path} is outside {base}"
ERR_YAML_PARSE = "Error parsing YAML file: {error}"
ERR_YAML_NOT_DICT = "YAML file must contain a dictionary."

# Pacemaker Defaults
DEFAULT_DELTA_SPLINE_BINS = 100
DEFAULT_EVALUATOR = "tensorpot"
DEFAULT_DISPLAY_STEP = 50
DEFAULT_MAX_FRAMES_ELEMENT_DETECTION = 10

# DFT Defaults
DEFAULT_DFT_MIXING_BETA = 0.7
DEFAULT_DFT_SMEARING_TYPE = "mv"
DEFAULT_DFT_SMEARING_WIDTH = 0.1
DEFAULT_DFT_DIAGONALIZATION = "david"
DEFAULT_DFT_MIXING_BETA_FACTOR = 0.5
DEFAULT_DFT_SMEARING_WIDTH_FACTOR = 2.0

# Training Defaults
DEFAULT_TRAINING_MAX_ITERATIONS = 1000
DEFAULT_TRAINING_BATCH_SIZE = 10
DEFAULT_PACEMAKER_NDENSITY = 2
DEFAULT_PACEMAKER_MAX_DEG = 6
DEFAULT_PACEMAKER_R0 = 1.5
DEFAULT_PACEMAKER_LOSS_KAPPA = 0.3
DEFAULT_PACEMAKER_LOSS_L1 = 1e-8
DEFAULT_PACEMAKER_LOSS_L2 = 1e-8
DEFAULT_PACEMAKER_REPULSION_SIGMA = 0.05
DEFAULT_PACEMAKER_OPTIMIZER = "BFGS"
DEFAULT_PACEMAKER_EMBEDDING_TYPE = "FinnisSinclair"
DEFAULT_PACEMAKER_RAD_BASE = "Chebyshev"

# OTF Defaults
DEFAULT_OTF_UNCERTAINTY_THRESHOLD = 5.0
DEFAULT_OTF_LOCAL_N_CANDIDATES = 20
DEFAULT_OTF_LOCAL_N_SELECT = 5
DEFAULT_OTF_MAX_RETRIES = 3

# MD Defaults
DEFAULT_MD_THERMO_FREQ = 10
DEFAULT_MD_DUMP_FREQ = 100
DEFAULT_MD_NEIGHBOR_SKIN = 2.0
DEFAULT_MD_ATOM_STYLE = "atomic"
DEFAULT_MD_TDAMP_FACTOR = 100.0
DEFAULT_MD_PDAMP_FACTOR = 1000.0
DEFAULT_MD_BASE_ENERGY = -100.0
DEFAULT_MD_CHECK_INTERVAL = 10
DEFAULT_MD_HYBRID_ZBL_INNER = 2.0
DEFAULT_MD_HYBRID_ZBL_OUTER = 2.5
MAX_MD_PRESSURE = 1.0e6
MAX_MD_DURATION = 1.0e6  # ps

# MC Defaults
DEFAULT_MC_SEED = 12345

# Validation Defaults
DEFAULT_VALIDATION_PHONON_SUPERCELL = [2, 2, 2]
DEFAULT_VALIDATION_PHONON_DISPLACEMENT = 0.01
DEFAULT_VALIDATION_PHONON_IMAGINARY_TOL = -0.05
DEFAULT_VALIDATION_ELASTIC_STRAIN = 0.01
DEFAULT_VALIDATION_ELASTIC_STEPS = 5

# Security constants
# Audit fix: Expanded list of dangerous characters
DANGEROUS_PATH_CHARS: Final[set[str]] = {
    ";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r", "\t", "?",
    "*", "[", "]", "{", "}", "'", '"', "!", "#",
}

# RAM Disk logic
_ram_disk_candidate = "/dev/shm"  # noqa: S108
DEFAULT_RAM_DISK_PATH = (
    _ram_disk_candidate if Path(_ram_disk_candidate).exists() else tempfile.gettempdir()
)

# MD Minimize defaults
DEFAULT_MD_MINIMIZE_FTOL = 1e-6
DEFAULT_MD_MINIMIZE_TOL = 1e-4
LAMMPS_MINIMIZE_MAX_ITER = 10000
LAMMPS_MINIMIZE_STEPS = 10000
LAMMPS_VELOCITY_SEED = 12345
# Allowed characters in LAMMPS commands: Alphanumeric, space, common punctuation including *
LAMMPS_SAFE_CMD_PATTERN = r"^[a-zA-Z0-9\s_\-\.\/=\"\*]+$"
LAMMPS_SCREEN_ARG = "-screen"
LAMMPS_MIN_STYLE_CG = "cg"

# Delta Learning
DEFAULT_LJ_PARAMS: Final[dict[str, float]] = {"sigma": 2.5, "epsilon": 1.0, "cutoff": 5.0}
FALLBACK_LJ_PARAMS: Final[dict[str, float]] = {"sigma": 2.0, "epsilon": 0.5, "cutoff": 4.0}

# Embedding
EMBEDDING_TOLERANCE_CELL = 0.1

# Errors
ERR_M3GNET_PRED_FAIL = "M3GNet prediction failed: {error}"
ERR_GEN_BASE_FAIL = "Base generator failed: {error}"
ERR_GEN_NCAND_NEG = "Number of candidates must be positive."
ERR_ORACLE_FAILED = "Oracle calculation failed: {error}"
ERR_ORACLE_ITERATOR = "Oracle failed to create iterator."
ERR_ORACLE_WARN_EMPTY = "Oracle received empty iterator. No calculations performed."
ERR_SIM_EXEC_FAIL = "Simulation execution failed: {error}"
ERR_SIM_SECURITY_FAIL = "Simulation security check failed: {error}"
ERR_SIM_SETUP_FAIL = "Simulation setup failed: {error}"
ERR_SIM_UNEXPECTED = "Unexpected simulation error: {error}"
ERR_STRUCTURE_NONE = "Structure cannot be None."
ERR_POTENTIAL_NOT_FOUND = "Potential file not found: {path}"
ERR_VAL_POT_NONE = "Validator requires a potential."
ERR_VAL_POT_NOT_FILE = "Potential path is not a file: {path}"
ERR_VAL_POT_OUTSIDE = "Potential path is outside allowed directory: {path}"
ERR_VAL_REQ_STRUCT = "Validator requires a structure."
ERR_VAL_STRUCT_EMPTY = "Structure is empty."
ERR_VAL_STRUCT_NONE = "Structure is None."
ERR_VAL_STRUCT_TYPE = "Invalid structure type: {type}"
ERR_VAL_STRUCT_VOL_FAIL = "Failed to compute structure volume: {error}"
ERR_VAL_STRUCT_ZERO_VOL = "Structure has near-zero or negative volume."
ERR_VAL_STRUCT_NAN_POS = "Structure contains non-finite atomic positions."
ERR_VAL_STRUCT_UNKNOWN_SYM = "Structure contains unknown chemical symbol: {symbol}"
ERR_VAL_STRUCT_DUMMY_ELEM = "Structure contains dummy element: {symbol} (Z=0)"

# DFT
RECIPROCAL_FACTOR = 2.0 * 3.141592653589793  # 2*PI approx

# Policy
DEFAULT_STRAIN_RANGE: Final[tuple[float, float]] = (-0.05, 0.05)

# Workflow modes
WORKFLOW_MODE_LEGACY = "legacy"
WORKFLOW_MODE_DISTILLATION = "distillation"

# Distillation steps
LOG_STEP_1 = "Step 1: DIRECT Sampling (Entropy Maximization)"
LOG_STEP_2 = "Step 2: MACE Uncertainty-based Active Learning"
LOG_STEP_3 = "Step 3: MACE Fine-tuning"
LOG_STEP_4 = "Step 4: Surrogate Data Generation"
LOG_STEP_5 = "Step 5: Surrogate Labeling"
LOG_STEP_6 = "Step 6: Pacemaker Base Training"
LOG_STEP_7 = "Step 7: Delta Learning (Fine-tuning with DFT)"

# PACE Driver script template (uses environment variables for security)
PACE_DRIVER_TEMPLATE = """
import sys
import os
import re
import numpy as np
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS

# Read potential path from environment for security
POTENTIAL_PATH = os.environ.get("PACE_POTENTIAL_PATH")
if not POTENTIAL_PATH:
    sys.stderr.write("Error: PACE_POTENTIAL_PATH not set\\n")
    sys.exit(1)

# Security: Validate POTENTIAL_PATH to prevent injection in pair_coeff
# Allow alphanumeric, dot, underscore, dash, slash.
if not re.match(r"^[a-zA-Z0-9_\\-\\.\\/]+$", POTENTIAL_PATH):
    sys.stderr.write("Error: Invalid characters in potential path\\n")
    sys.exit(1)

if not os.path.exists(POTENTIAL_PATH):
    sys.stderr.write(f"Error: Potential file not found at {POTENTIAL_PATH}\\n")
    sys.exit(1)

def read_input():
    try:
        lines = sys.stdin.readlines()
        if not lines:
            return None, None, None

        # Parse EON format (assumed based on standard command line potentials)
        # 1. Number of atoms
        num_atoms = int(lines[0].strip())

        # 2. Box (3 lines)
        cell = np.zeros((3, 3))
        cell[0] = [float(x) for x in lines[1].split()]
        cell[1] = [float(x) for x in lines[2].split()]
        cell[2] = [float(x) for x in lines[3].split()]

        # 3. Coordinates (N lines)
        # EON usually sends only coordinates x y z, not species.
        # We need to map them to species from a template.
        coords = []
        for i in range(num_atoms):
            coords.append([float(x) for x in lines[4+i].split()])

        return num_atoms, cell, np.array(coords)
    except Exception as e:
        sys.stderr.write(f"Error reading input: {e}\\n")
        sys.exit(1)

def main():
    try:
        # 1. Load template structure (pos.con) to get species
        if not os.path.exists("pos.con"):
            sys.stderr.write("Error: pos.con not found\\n")
            sys.exit(1)

        try:
            # EON uses .con format, ASE supports it
            template = read("pos.con", format="eon")
        except Exception:
            # Fallback if ASE cannot read eon format directly or extension issue
            # Try reading as 'con' or just assume it works if supported
            template = read("pos.con")

        # 2. Read current configuration from stdin
        n, cell, coords = read_input()
        if n is None:
            sys.exit(0)

        if n != len(template):
            sys.stderr.write(f"Error: Atom count mismatch ({n} vs {len(template)})\\n")
            sys.exit(1)

        # 3. Update structure
        template.set_cell(cell)
        template.set_positions(coords)

        # 4. Setup Calculator
        # Using LAMMPS via ASE is robust.
        # For speed, one might use lammps python module directly,
        # but constructing the input script for PACE is complex.
        # We rely on ASE's LAMMPS calculator or similar.
        # We need to specify the potential file.

        # We construct a pair_style command for PACE.
        # Assuming 'pace' pair style is available in the LAMMPS binary.
        # pair_style pace
        # pair_coeff * * potential.yace Element1 Element2 ...

        species = sorted(list(set(template.get_chemical_symbols())))
        species_str = " ".join(species)

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {POTENTIAL_PATH} {species_str}"]
        }

        calc = LAMMPS(parameters=parameters, files=[POTENTIAL_PATH])
        template.calc = calc

        # 5. Compute
        energy = template.get_potential_energy()
        forces = template.get_forces()

        # 6. Output results
        # Format: Energy (1 line)
        # Forces (N lines, x y z)
        print(f"{energy:.16f}")
        for f in forces:
            print(f"{f[0]:.16f} {f[1]:.16f} {f[2]:.16f}")

    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
