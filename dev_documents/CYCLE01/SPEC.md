# Cycle 01 Specification: Core Framework & DIRECT Sampling

## 1. Summary
This cycle lays the foundational bedrock for the **PyAceMaker** system. The primary objective is to establish the project structure, define the core domain models that will ensure type safety throughout the lifecycle, and implement the first step of the 7-Step Workflow: **DIRECT Sampling**.

We will implement the `Orchestrator` skeleton, which is responsible for reading the configuration and dispatching tasks. The `structure_generator` module will be initialized with the `DirectSampler`, capable of generating diverse atomic structures by maximizing entropy in the descriptor space (or random packing in this initial phase). This eliminates the need for manual selection of initial structures.

## 2. System Architecture

The following file structure will be created/updated. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── **pyproject.toml**              # (Already created)
├── **config.yaml**                 # Default configuration template
├── src/
│   └── pyacemaker/
│       ├── **__init__.py**
│       ├── **main.py**             # CLI Entry point
│       ├── **orchestrator.py**     # Main workflow controller
│       ├── **logger.py**           # Logging setup
│       ├── core/
│       │   ├── **__init__.py**
│       │   ├── **base.py**         # Abstract Base Classes (BaseGenerator, BaseOracle)
│       │   └── **exceptions.py**   # Custom exceptions
│       ├── domain_models/
│       │   ├── **__init__.py**
│       │   ├── **config.py**       # Pydantic Config Models (PyAceConfig)
│       │   ├── **structure.py**    # Structure Config (StructureConfig)
│       │   ├── **data.py**         # AtomStructure & Dataset Models
│       │   └── **state.py**        # Workflow State Models
│       ├── structure_generator/
│       │   ├── **__init__.py**
│       │   └── **direct.py**       # DIRECT/Random sampling implementation
│       └── utils/
│           ├── **__init__.py**
│           ├── **io.py**           # XYZ/YAML I/O utilities
└── tests/
    ├── **conftest.py**
    ├── unit/
    │   ├── **test_config.py**
    │   ├── **test_domain_models_data.py**
    │   ├── **test_domain_models_state.py**
    │   └── **test_structure_generator_direct.py**
    └── integration/
        └── **test_cycle01_integration.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)
We use **Pydantic** to enforce strict typing and validation.

*   **`structure.py`**:
    *   `StructureConfig`:
        *   `elements`: List[str] (e.g., ["C", "H"])
        *   `num_structures`: int ( > 0)
        *   `r_cut`: float (cutoff radius, default 2.0)
        *   `supercell_size`: List[int] (e.g. [1, 1, 1])

*   **`config.py`**:
    *   `PyAceConfig`: The root object.
        *   `structure`: StructureConfig
        *   `workflow`: WorkflowConfig
        *   `logging`: LoggingConfig
        # Other configs (DFT, MD, Training) are present but optional/placeholder for Cycle 01

*   **`data.py`**:
    *   `AtomStructure`:
        *   `atoms`: Arbitrary type (ASE Atoms).
        *   `energy`: Optional[float]
        *   `forces`: Optional[List[List[float]]]
        *   `uncertainty`: Optional[float]
        *   `provenance`: str (e.g., "DIRECT_SAMPLING")
        *   `metadata`: Dict[str, Any]

*   **`state.py`**:
    *   `WorkflowStatus`: Enum (PENDING, RUNNING, COMPLETED, FAILED)
    *   `StepState`: Tracks status of individual steps.
    *   `GlobalState`: Singleton-like object to save/load progress (WorkflowStatus, current_step).

### 3.2. Interfaces (`core/base.py`)
Abstract Base Classes (ABCs) to decouple implementation from logic.

*   **`BaseGenerator`**:
    *   `generate(self, n_candidates: int) -> Iterator[Atoms]`

### 3.3. DIRECT Sampler (`structure_generator/direct.py`)
This class implements the sampling strategy.
*   **Logic**: For Cycle 01, it implements a random structure generation with hard-sphere constraints.
*   **Algorithm**: Randomly places atoms in a box defined by `supercell_size` (or default unit cell). Rejects placements that violate `r_cut` distance to existing atoms.
*   **Output**: An iterator of `Atoms` objects.

### 3.4. Orchestrator (`orchestrator.py`)
The central hub. In Cycle 01, it implements:
1.  `initialize_workspace()`: Creates directories, initializes state.
2.  `run_step1()`: Instantiates `DirectSampler`, calls `generate`, saves output to `step1_initial.xyz`.

## 4. Implementation Approach

### Step 4.1: Project Skeleton & Utilities
1.  Ensure directory tree exists.
2.  Implement `logger.py` (already exists, verify).
3.  Implement `utils/io.py` for reading/writing YAML and XYZ files.

### Step 4.2: Domain Models
1.  Update `structure.py` with `r_cut`.
2.  Implement `data.py` (`AtomStructure`).
3.  Implement `state.py` (`GlobalState`, `WorkflowStatus`).

### Step 4.3: Structure Generator
1.  Implement `DirectSampler` in `structure_generator/direct.py`.
2.  Use `ase.build` methods or custom logic for random packing.

### Step 4.4: Orchestrator & CLI
1.  Implement `Orchestrator.run_step1()`.
2.  Implement `main.py` CLI commands.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_domain_models_data.py`**: Verify `AtomStructure` validation.
*   **`test_domain_models_state.py`**: Verify state transitions and serialization.
*   **`test_structure_generator_direct.py`**:
    *   Call `DirectSampler.generate(n=10)`.
    *   Assert 10 structures are returned.
    *   Assert all structures have no overlapping atoms (minimum distance check > r_cut).

### 5.2. Integration Testing
*   **`test_cycle01_integration.py`**:
    *   Create a temporary `config.yaml`.
    *   Run `Orchestrator` to execute Step 1.
    *   Check that `step1_initial.xyz` exists and contains valid ASE atoms.
    *   Verify the `workflow_state.json` is updated to mark Step 1 as COMPLETED.
