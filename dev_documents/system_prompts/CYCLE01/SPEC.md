# Cycle 01 Specification: Core Framework & DIRECT Sampling

## 1. Summary
This cycle lays the foundational bedrock for the **PyAceMaker** system. The primary objective is to establish the project structure, define the core domain models that will ensure type safety throughout the lifecycle, and implement the first step of the 7-Step Workflow: **DIRECT Sampling**.

We will implement the `Orchestrator` skeleton, which is responsible for reading the configuration and dispatching tasks. The `structure_generator` module will be initialized with the `DirectSampler`, capable of generating diverse atomic structures by maximizing entropy in the descriptor space. This eliminates the need for manual selection of initial structures. We will also implement a "Mock Oracle" to verify the data flow without requiring external physics codes.

## 2. System Architecture

The following file structure will be created. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── **pyproject.toml**              # (Already created)
├── **config.yaml**                 # Default configuration template
├── src/
│   └── pyacemaker/
│       ├── **__init__.py**
│       ├── **main.py**             # CLI Entry point
│       ├── **orchestrator.py**     # Main workflow controller (Skeleton)
│       ├── **constants.py**        # Global constants
│       ├── core/
│       │   ├── **__init__.py**
│       │   ├── **interfaces.py**   # Abstract Base Classes (IOracle, IGenerator)
│       │   └── **exceptions.py**   # Custom exceptions
│       ├── domain_models/
│       │   ├── **__init__.py**
│       │   ├── **config.py**       # Pydantic Config Models
│       │   ├── **data.py**         # AtomStructure & Dataset Models
│       │   └── **state.py**        # Workflow State Models
│       ├── structure_generator/
│       │   ├── **__init__.py**
│       │   └── **direct.py**       # DIRECT sampling implementation
│       ├── oracle/
│       │   ├── **__init__.py**
│       │   └── **mock.py**         # Lennard-Jones Mock Oracle
│       └── utils/
│           ├── **__init__.py**
│           ├── **io.py**           # XYZ/YAML I/O utilities
│           └── **logging.py**      # Logging configuration
└── tests/
    ├── **conftest.py**
    ├── unit/
    │   ├── **test_config.py**
    │   └── **test_direct_sampling.py**
    └── integration/
        └── **test_pipeline_step1.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)
We use **Pydantic** to enforce strict typing and validation.

*   **`config.py`**:
    *   `DirectSamplingConfig`:
        *   `target_points`: int ( > 0)
        *   `elements`: List[str] (e.g., ["C", "H"])
        *   `r_cut`: float (cutoff radius)
    *   `GlobalConfig`: The root object.
        *   `work_dir`: Path (default: "./work")
        *   `random_seed`: int
        *   `sampling`: DirectSamplingConfig

*   **`data.py`**:
    *   `AtomStructure`:
        *   `atoms`: Arbitrary type (to be serialized/deserialized from ASE Atoms). *Note: Pydantic doesn't validate ASE objects natively, so we'll use a wrapper or `arbitrary_types_allowed`.*
        *   `energy`: Optional[float]
        *   `forces`: Optional[List[List[float]]]
        *   `uncertainty`: Optional[float]
        *   `provenance`: str (e.g., "DIRECT_SAMPLING")
        *   `metadata`: Dict[str, Any]

*   **`state.py`**:
    *   `WorkflowStatus`: Enum (PENDING, RUNNING, COMPLETED, FAILED)
    *   `StepState`: Tracks status of Step 1-7.
    *   `GlobalState`: Singleton-like object to save/load progress.

### 3.2. Interfaces (`core/interfaces.py`)
Abstract Base Classes (ABCs) to decouple implementation from logic.

*   **`IGenerator`**:
    *   `generate(self, config: GenerationConfig) -> List[AtomStructure]`
*   **`IOracle`**:
    *   `compute(self, structure: AtomStructure) -> AtomStructure` (Returns structure with Energy/Forces filled)

### 3.3. DIRECT Sampler (`structure_generator/direct.py`)
This class implements the "Entropy Maximization" strategy.
*   **Logic**: It constructs a simplified descriptor space (e.g., pair distances or a simple fingerprint).
*   **Algorithm**: It uses a pseudo-random or quasi-random sequence (like Sobol) filtered by a "maximin" criterion—selecting points that maximize the minimum distance to existing points in the descriptor space. This ensures the initial pool covers the widest possible phase space.
*   **Output**: A list of `AtomStructure` objects with random cell sizes and atomic positions, subject to hard-sphere constraints (atoms cannot overlap).

### 3.4. Orchestrator (`orchestrator.py`)
The central hub. In Cycle 01, it implements:
1.  `load_config(path)`: Parses YAML into `GlobalConfig`.
2.  `initialize_workspace()`: Creates directories.
3.  `run_step1()`: Instantiates `DirectSampler`, calls `generate`, saves output to `work_dir/step1_initial.xyz`.

## 4. Implementation Approach

### Step 4.1: Project Skeleton & Utilities
1.  Create the directory tree.
2.  Implement `utils/logging.py` to set up colored console output and file logging.
3.  Implement `utils/io.py` for reading/writing YAML and XYZ files (wrapping `ase.io`).

### Step 4.2: Domain Models
1.  Define `config.py` with Pydantic. Add validators to ensure `r_cut` is positive and `elements` are valid periodic table symbols.
2.  Define `data.py` to hold the structure data.

### Step 4.3: Core Interfaces
1.  Define `IGenerator` and `IOracle` in `core/interfaces.py`.

### Step 4.4: Structure Generator
1.  Implement `DirectSampler`.
2.  Use `ase.build` methods to create random bulk/molecule structures.
3.  Implement a "hard-sphere check" to prevent atom overlap.
4.  Implement the "Maximin" selection logic.

### Step 4.5: Orchestrator & CLI
1.  Implement `Orchestrator.run_step1()`.
2.  Implement `main.py` using `argparse` to accept `--config` and `--step`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_config.py`**:
    *   Load a valid YAML and assert fields are correct.
    *   Load an invalid YAML (missing fields, negative cutoff) and assert `ValidationError` is raised.
*   **`test_direct_sampling.py`**:
    *   Call `DirectSampler.generate(n=10)`.
    *   Assert 10 structures are returned.
    *   Assert all structures have no overlapping atoms (minimum distance check).
    *   Assert structures contain the correct element types.

### 5.2. Integration Testing
*   **`test_pipeline_step1.py`**:
    *   Create a temporary `config.yaml`.
    *   Run `Orchestrator` to execute Step 1.
    *   Check that `work_dir` is created.
    *   Check that `step1_initial.xyz` exists and contains valid ASE atoms.
    *   Verify the `workflow_state.json` is updated to mark Step 1 as COMPLETED.
