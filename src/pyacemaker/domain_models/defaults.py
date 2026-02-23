# Configuration Defaults
DEFAULT_STATE_FILE = "state.json"
DEFAULT_DATA_DIR = "data"
DEFAULT_ACTIVE_LEARNING_DIR = "active_learning"
DEFAULT_POTENTIALS_DIR = "potentials"
DEFAULT_BATCH_SIZE = 5
DEFAULT_N_CANDIDATES = 10
DEFAULT_CHECKPOINT_INTERVAL = 1

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
