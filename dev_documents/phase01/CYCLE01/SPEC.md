# Cycle 01 Specification: Foundation & Configuration

## 1. Summary
This cycle establishes the foundational architecture of the **PYACEMAKER** system. The primary goal is to set up the project repository, define the robust configuration management system using **Pydantic**, and implement the skeleton of the central **Orchestrator**. We will not run any physics simulations in this cycle; instead, we focus on ensuring that the system can correctly parse user inputs, validate configuration constraints (e.g., preventing negative temperatures), and initialize the logging infrastructure. This forms the "spine" upon which all subsequent physical modules will be attached.

## 2. System Architecture
The file structure below highlights the files to be created or modified in **bold**.

```ascii
pyacemaker/
├── pyproject.toml
├── README.md
├── **config.yaml**               # Example configuration file
├── src/
│   └── pyacemaker/
│       ├── **__init__.py**
│       ├── **main.py**           # Entry point (CLI)
│       ├── **config.py**         # Pydantic Schemas
│       ├── **logger.py**         # Logging Configuration
│       ├── **orchestrator.py**   # The Brain (Skeleton)
│       ├── core/
│       │   ├── **__init__.py**
│       │   ├── **base.py**       # Abstract Base Classes
│       │   ├── generator.py      # (Placeholder)
│       │   ├── oracle.py         # (Placeholder)
│       │   ├── trainer.py        # (Placeholder)
│       │   └── engine.py         # (Placeholder)
│       └── utils/
│           └── **io.py**         # YAML/JSON helpers
└── tests/
    ├── **__init__.py**
    ├── **test_config.py**
    └── **test_orchestrator.py**
```

## 3. Design Architecture

### 3.1 Configuration Data Model (`config.py`)
We leverage **Pydantic V2** for strict type validation. The configuration is hierarchical:

*   **`PyAceConfig` (Root)**:
    *   `project_name`: str
    *   `structure`: `StructureConfig`
    *   `dft`: `DFTConfig`
    *   `training`: `TrainingConfig`
    *   `md`: `MDConfig`
    *   `workflow`: `WorkflowConfig`

*   **Key Constraints**:
    *   Paths must be validated to exist (or be creatable).
    *   Numerical values (cutoffs, temperatures) must be positive.
    *   Enums must be used for discrete choices (e.g., `optimizer: "bfgs" | "cg"`).

### 3.2 Orchestrator Design (`orchestrator.py`)
The `Orchestrator` class follows the **Singleton** or **Context Manager** pattern to manage the application lifecycle.
*   **Attributes**: Holds instances of the abstract base classes (`BaseGenerator`, `BaseOracle`, etc.).
*   **Methods**: `load_config()`, `initialize_modules()`, `run()`.

### 3.3 Abstract Base Classes (`core/base.py`)
To ensure modularity and testability (mocking), we define:
*   `BaseGenerator`: Interface for structure generation.
*   `BaseOracle`: Interface for energy/force calculation.
*   `BaseTrainer`: Interface for potential fitting.
*   `BaseEngine`: Interface for MD simulations.

## 4. Implementation Approach

### Step 1: Repository & Environment Setup
*   Initialize the project structure.
*   Configure `ruff` and `mypy` as per `pyproject.toml`.
*   Create the `src/pyacemaker` package.

### Step 2: Configuration Module
*   Implement `src/pyacemaker/config.py`.
*   Define all Pydantic models.
*   Implement `src/pyacemaker/utils/io.py` to load YAML into these models.

### Step 3: Logging Infrastructure
*   Implement `src/pyacemaker/logger.py`.
*   Configure structured logging (console output + file rotation).

### Step 4: Core Abstractions & Orchestrator
*   Define ABCs in `src/pyacemaker/core/base.py`.
*   Implement the `Orchestrator` class in `src/pyacemaker/orchestrator.py` that accepts a `PyAceConfig` object.
*   Create the `main.py` CLI using `argparse` or `typer` to accept a config file path.

## 5. Test Strategy

### 5.1 Unit Testing (`test_config.py`)
*   **Valid Config**: Create a full `config.yaml` and assert it parses correctly into Pydantic models.
*   **Invalid Config**: Deliberately introduce errors (e.g., `cutoff: -5.0`, `missing_field`) and assert that `ValidationError` is raised with a meaningful message.
*   **Type Coercion**: specific string-to-number conversions should be tested.

### 5.2 Integration Testing (`test_orchestrator.py`)
*   **Initialization**: Instantiate the `Orchestrator` with a valid config. Verify that it initializes its internal state (iteration counter = 0) and logging system.
*   **Mocking**: Since core modules are placeholders, the test should verify that the Orchestrator holds `None` or Mock objects for them at this stage.

### 5.3 Coverage Goals
*   100% coverage on `config.py` logic (validators).
*   Verify that `main.py` correctly handles `FileNotFoundError`.
