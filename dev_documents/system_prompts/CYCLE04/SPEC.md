# Cycle 04 Specification: Surrogate Labeling & Base ACE Training

## 1. Summary
This cycle implements **Steps 5 and 6** of the MACE Distillation Workflow. Having generated a large, physically relevant set of structures via MD in Cycle 03, we now need to assign energies and forces to them.

First, we utilize the fine-tuned MACE model to "label" these thousands of structures. This creates a massive synthetic dataset. Since MACE is orders of magnitude faster than DFT, this step is computationally cheap yet provides a high-fidelity representation of the MACE potential energy surface.

Second, we train the **Base ACE Potential**. We wrap the `pacemaker` library to train an Atomic Cluster Expansion potential on this synthetic dataset. This results in an extremely fast interatomic potential that mimics the MACE model's behavior, serving as a solid foundation for the final Delta Learning step.

## 2. System Architecture

The following file structure will be created or modified. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── **orchestrator.py**     # Updated to handle Steps 5 & 6
│       ├── domain_models/
│       │   ├── **config.py**       # Added AceTrainingConfig
│       ├── trainer/
│       │   ├── **ace_trainer.py**  # Pacemaker Wrapper
│       ├── oracle/
│       │   └── **mace_wrapper.py** # Updated with batch_compute
│       └── utils/
│           └── **io.py**           # Added conversion to Pacemaker formats
└── tests/
    ├── unit/
    │   ├── **test_ace_trainer.py**
    │   └── **test_data_conversion.py**
    └── integration/
        └── **test_pipeline_step5_6.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)

*   **`AceTrainingConfig`**:
    *   `cutoff`: float (e.g., 5.0 Angstrom)
    *   `elements`: List[str]
    *   `ladder_step`: List[int] (Body order complexity)
    *   `ladder_type`: str (e.g., "power_order")
    *   `max_basis_grade`: int

### 3.2. MACE Wrapper Upgrade (`oracle/mace_wrapper.py`)

*   **`MaceSurrogate.compute_batch(structures)`**:
    *   **Goal**: Optimize inference for large datasets.
    *   **Implementation**: Use `mace-torch`'s batch processing capabilities (collate functions) to process chunks of atoms (e.g., 100 frames at a time) on the GPU, rather than a loop of single-frame calls.

### 3.3. ACE Trainer (`trainer/ace_trainer.py`)

*   **`PacemakerWrapper(ITrainer)`**:
    *   **Responsibilities**:
        1.  **Data Conversion**: Convert `step4_surrogate_data.xyz` (ASE format) into the specific `.pdata` or extended `.xyz` format required by Pacemaker, ensuring `energy`, `forces`, and `virial` are correctly mapped.
        2.  **Config Generation**: Generate the `input.yaml` required by `pacemaker`, translating `AceTrainingConfig` into Pacemaker's syntax.
        3.  **Execution**: Invoke `pacemaker` (e.g., via `subprocess` calling `pacemaker input.yaml`).
        4.  **Artifact Management**: Locate the resulting `potential.yace` and move it to `work_dir/models/`.

### 3.4. Orchestrator Update

*   **`run_step5()`**:
    1.  Load `step4_surrogate_data.xyz` (Unlabeled).
    2.  Load `MaceSurrogate` (Fine-tuned).
    3.  Call `compute_batch` to label all structures.
    4.  Save to `step5_surrogate_labeled.xyz`.

*   **`run_step6()`**:
    1.  Initialize `PacemakerWrapper`.
    2.  Train ACE model using `step5_surrogate_labeled.xyz`.
    3.  Save output to `work_dir/models/ace_base.yace`.

## 4. Implementation Approach

### Step 4.1: Batch Labeling
1.  Implement `compute_batch` in `MaceSurrogate`.
2.  Ensure memory safety: stream the dataset or process in chunks if it's too large for RAM.

### Step 4.2: Data Conversion
1.  Implement a utility in `utils/io.py` to write "ExtXYZ" format compatible with Pacemaker.
2.  Key requirement: The header must define columns like `energy:R:1`, `forces:R:3`.

### Step 4.3: Pacemaker Integration
1.  Implement `PacemakerWrapper`.
2.  Use `yaml` to write the `input.yaml`.
3.  Support a "Dry Run" or "Mock Mode" where a dummy `.yace` file is created if `pacemaker` is not installed (crucial for basic CI environments).

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_data_conversion.py`**:
    *   Create an ASE Atoms object with energy/forces.
    *   Convert to Pacemaker format string.
    *   Assert headers are correct and data is formatted (e.g., correct number of columns).
*   **`test_ace_trainer.py`**:
    *   Verify `input.yaml` generation matches the `AceTrainingConfig`.
    *   Test the command construction logic.

### 5.2. Integration Testing
*   **`test_pipeline_step5_6.py`**:
    *   **Prerequisite**: `step4_surrogate_data.xyz` (can be dummy data).
    *   **Step 5**: Run labeling. Verify output XYZ has `energy` and `forces` properties.
    *   **Step 6**: Run ACE training (Mock). Verify `ace_base.yace` is created.
