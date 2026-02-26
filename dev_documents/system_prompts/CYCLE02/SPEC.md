# Cycle 02 Specification: MACE Oracle & Active Learning

## 1. Summary
In this cycle, we implement the "Brain" of PyAceMaker: the **MACE Surrogate Oracle** and the **Active Learning** module. This corresponds to **Step 2** of the workflow. The goal is to intelligently select which structures from the initial pool (generated in Cycle 01) require expensive DFT calculations.

We will implement a wrapper around the `mace-torch` library to load pre-trained models and compute not just energy and forces, but also **uncertainty**. We will then implement the `UncertaintyFilter`, which ranks structures based on their uncertainty score and selects the most informative ones. Finally, we will define the `DFTManager` interface and a Mock implementation, allowing the system to "label" these selected structures without needing a real VASP/QE installation yet.

## 2. System Architecture

The following file structure will be created or modified. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── **orchestrator.py**     # Updated to handle Step 2
│       ├── domain_models/
│       │   ├── **config.py**       # Added ActiveLearningConfig
│       │   └── **data.py**         # Updated to support "Labeled" status
│       ├── oracle/
│       │   ├── **mace_wrapper.py** # MACE Model Wrapper
│       │   ├── **uncertainty.py**  # Uncertainty quantification logic
│       │   └── **dft_manager.py**  # Abstract & Mock DFT Manager
└── tests/
    ├── unit/
    │   ├── **test_uncertainty.py**
    │   └── **test_mace_wrapper.py**
    └── integration/
        └── **test_pipeline_step2.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)

*   **`ActiveLearningConfig`**:
    *   `model_path`: Path (Path to the MACE model checkpoint, e.g., "MACE-MP-0.model")
    *   `n_selection`: int (Number of structures to select for DFT)
    *   `uncertainty_method`: Enum ("variance", "committee", "force_ensemble")
    *   `dft_calculator`: str (e.g., "VASP", "MOCK")

### 3.2. MACE Wrapper (`oracle/mace_wrapper.py`)

*   **`MaceSurrogate(IOracle)`**:
    *   **Responsibilities**: Load the PyTorch model. Convert `AtomStructure` to MACE input tensors.
    *   **Methods**:
        *   `compute_uncertainty(structure)`: Returns a scalar uncertainty score.
        *   `compute_energy_forces(structure)`: Returns (E, F) predictions.
    *   **Uncertainty Logic**: If the model is an ensemble, compute variance. If single model, use last-layer variance or similar proxy if supported, otherwise return random (for testing) or raise NotImplemented.

### 3.3. DFT Manager (`oracle/dft_manager.py`)

*   **`IDFTManager(ABC)`**:
    *   `submit_batch(structures: List[AtomStructure]) -> JobID`
    *   `check_status(job_id) -> JobStatus`
    *   `retrieve_results(job_id) -> List[LabeledStructure]`
*   **`MockDFTManager`**:
    *   Simulates a delay (sleep).
    *   Returns ground truth values using a simple potential (Lennard-Jones or EMT) + Gaussian noise to simulate "DFT Accuracy".

### 3.4. Orchestrator Update

*   **`run_step2()`**:
    1.  Load `step1_initial.xyz`.
    2.  Initialize `MaceSurrogate`.
    3.  Compute uncertainty for all structures.
    4.  Sort by uncertainty (Descending).
    5.  Select top `n_selection`.
    6.  Send to `DFTManager`.
    7.  Save selected (labeled) structures to `step2_active_set.xyz`.

## 4. Implementation Approach

### Step 4.1: MACE Integration
1.  Implement `MaceSurrogate` class. Use `mace.calculators.MACECalculator` internally if available.
2.  Implement robust error handling for model loading (handle missing file).
3.  **Critical**: For the purpose of this cycle, if `mace-torch` is not installed or model is missing, implement a "Mock Mode" that returns random uncertainty to allow development without heavy weights.

### Step 4.2: Uncertainty Logic
1.  Implement `UncertaintyCalculator`. Strategy:
    *   **Committee**: If multiple models are provided, `std(E)`.
    *   **Single Model**: Use `forces` magnitude variance or a specific MACE uncertainty output if available.

### Step 4.3: Orchestrator Logic
1.  Update `Orchestrator` to read Step 1 output.
2.  Implement the filtering loop.
3.  Ensure "Idempotency": If `step2_active_set.xyz` exists, skip calculation.

### Step 4.4: Mock DFT
1.  Implement `MockDFTManager` using `ase.calculators.emt` or `lj`.
2.  Ensure it preserves the `provenance` metadata.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_uncertainty.py`**:
    *   Create dummy structures.
    *   Pass to `MaceSurrogate` (mocked).
    *   Assert that structures with higher "simulated" variance get higher scores.
*   **`test_filtering.py`**:
    *   Given a list of 10 structures with known uncertainty scores.
    *   Request top 3.
    *   Assert the correct 3 IDs are returned.

### 5.2. Integration Testing
*   **`test_pipeline_step2.py`**:
    *   **Prerequisite**: Run Step 1 (or create dummy `step1.xyz`).
    *   **Action**: Run Orchestrator Step 2.
    *   **Verification**:
        *   `step2_active_set.xyz` exists.
        *   Count of atoms matches `n_selection` config.
        *   Structures in `step2` are a subset of `step1`.
        *   Structures have `energy` and `forces` populated (by Mock DFT).
