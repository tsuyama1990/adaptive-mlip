# Constants for PyAceMaker

# File names
FILENAME_CANDIDATES = "candidates.xyz"
FILENAME_TRAINING = "training_data.xyz"
FILENAME_POTENTIAL = "potential.yace"

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

# Template strings
TEMPLATE_ITER_DIR = "iter_{iteration:03d}"
TEMPLATE_POTENTIAL_FILE = "generation_{iteration:03d}.yace"

# Physics Constants
RECIPROCAL_FACTOR = 6.283185307179586  # 2 * pi

# Embedding Constants
EMBEDDING_TOLERANCE_CELL = 1e-6
EMBEDDING_TOLERANCE_LENGTH = 1e-9
