# Cycle 01: Core Infrastructure & Domain Models

## 1. Summary
Cycle 01 lays the foundation for the `pyacemaker` system. The primary goal is to establish the core data structures (Pydantic models) and abstract interfaces (ABCs) that will drive the entire MACE Knowledge Distillation pipeline. By enforcing strict typing and a clear separation of concerns from the start, we ensure that subsequent cycles (Sampling, Oracle, Training) can be developed in parallel with minimal friction. This cycle also introduces a functional `MockOracle` to enable end-to-end testing without external DFT dependencies.

## 2. System Architecture

### 2.1. File Structure
**Files to be created in this cycle are bolded.**

```text
src/
└── pyacemaker/
    ├── **__init__.py**
    ├── **main.py**                     # CLI Entrypoint (Skeleton)
    ├── **config.py**                   # Global Configuration Loading
    ├── **core/**
    │   ├── **__init__.py**
    │   ├── **base.py**                 # Abstract Base Classes (Generator, Oracle, Trainer)
    │   └── **orchestrator.py**         # Workflow Controller (Skeleton)
    ├── **domain_models/**
    │   ├── **__init__.py**
    │   ├── **config.py**               # Pydantic Config Schema
    │   ├── **data.py**                 # AtomStructure & Label Schemas
    │   └── **defaults.py**             # Default Configuration Values
    ├── **modules/**
    │   ├── **__init__.py**
    │   └── **mock_oracle.py**          # Mock Oracle Implementation
    └── **utils/**
        ├── **__init__.py**
        ├── **io.py**                   # File I/O (XYZ, LAMMPS)
        └── **logging.py**              # Centralized Logging
```

### 2.2. Component Interaction
*   **User** provides a `config.yaml`.
*   **`main.py`** loads the config via `domain_models.config.PyAceConfig`.
*   **`Orchestrator`** initializes the pipeline components based on the config.
*   **`MockOracle`** implements `core.base.BaseOracle` and returns dummy energy/forces for testing.

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)
We use Pydantic V2 for robust data validation.

#### `PyAceConfig`
The root configuration object.
*   **Fields:**
    *   `system`: `SystemConfig` (elements, composition)
    *   `dft`: `DftConfig` (code: VASP/QE/Mock, command, potential_path)
    *   `mace`: `MaceConfig` (model_path, device)
    *   `workflow`: `WorkflowConfig` (steps enabled, output_dir)
*   **Validation:**
    *   Ensure `elements` matches `composition`.
    *   Verify `potential_path` exists if not Mock.

#### `AtomStructure`
A wrapper around `ase.Atoms` to carry metadata through the pipeline.
*   **Fields:**
    *   `atoms`: `ase.Atoms` (The actual structure)
    *   `energy`: `float | None`
    *   `forces`: `np.ndarray | None`
    *   `stress`: `np.ndarray | None`
    *   `uncertainty`: `float | None`
    *   `provenance`: `dict` (e.g., `{'step': 'direct_sampling', 'method': 'random'}`)
*   **Methods:**
    *   `to_ase()`: Returns `ase.Atoms` with attached calculator results.
    *   `from_ase()`: Creates `AtomStructure` from `ase.Atoms`.

### 3.2. Core Interfaces (`core/base.py`)
Abstract Base Classes (ABCs) defining the contract for all modules.

#### `BaseGenerator`
*   **Abstract Methods:**
    *   `generate(n_structures: int) -> List[AtomStructure]`

#### `BaseOracle`
*   **Abstract Methods:**
    *   `compute(structure: AtomStructure) -> AtomStructure` (Returns structure with energy/forces populated)
    *   `compute_batch(structures: List[AtomStructure]) -> List[AtomStructure]`

#### `BaseTrainer`
*   **Abstract Methods:**
    *   `train(dataset: List[AtomStructure], **kwargs) -> Path` (Returns path to trained model)

## 4. Implementation Approach

1.  **Setup Environment:** Initialize `pyproject.toml` dependencies (`pydantic`, `ase`, `numpy`).
2.  **Define Models:** Implement `domain_models/data.py` first, as it is the primary data carrier. Then implement `domain_models/config.py`.
3.  **Define Interfaces:** Create `core/base.py` with `abc.ABC`.
4.  **Implement Mock:** Create `modules/mock_oracle.py` using a simple Lennard-Jones potential (via `ase.calculators.lj`).
5.  **Implement Utils:** Create `utils/io.py` for reading/writing XYZ files with extended metadata (using `ase.io` internally).
6.  **Skeleton CLI:** Create `main.py` that loads a config file, instantiates a `MockOracle`, computes energy for a dummy structure, and prints the result.

## 5. Test Strategy

### 5.1. Unit Tests (`tests/unit/`)
*   **Config Validation:**
    *   Test loading a valid `config.yaml`.
    *   Test loading an invalid `config.yaml` (missing fields, wrong types) and assert `ValidationError`.
*   **Data Models:**
    *   Test `AtomStructure` serialization/deserialization.
    *   Test `AtomStructure` integration with `ase.Atoms`.
*   **Mock Oracle:**
    *   Test `MockOracle.compute()` returns finite energy and forces.
    *   Test `MockOracle` respects the `elements` defined in config.

### 5.2. Integration Tests (`tests/integration/`)
*   **End-to-End Config Loading:** Verify that `main.py` can parse a full production config file without error.
*   **Pipeline Data Flow:** Create a test that:
    1.  Instantiates `AtomStructure`.
    2.  Passes it to `MockOracle`.
    3.  Writes the result to disk via `utils.io`.
    4.  Reads it back and asserts data integrity.
