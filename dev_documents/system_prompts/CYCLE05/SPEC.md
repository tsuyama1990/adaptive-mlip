# Cycle 05 Specification: Delta Learning

## 1. Summary
This cycle implements **Step 7**, the final and most critical stage of the workflow: **Delta Learning**.

Up to this point, we have built a Base ACE potential that mimics the MACE surrogate. However, MACE itself is an approximation. To achieve true first-principles accuracy, we must correct the systematic errors of the MACE model using the high-precision DFT data collected in Step 2.

In this cycle, we implement the `DeltaTrainer`. This module fine-tunes the Base ACE potential using the "Active Set" (DFT data). The key technical challenge is to balance the large volume of surrogate data (which provides topology and stability) with the small volume of DFT data (which provides accuracy). We achieve this by configuring the loss function weights in Pacemaker, heavily penalizing errors on the DFT structures while keeping the surrogate data as a regularizer (or using the Base Model as a prior).

## 2. System Architecture

The following file structure will be created or modified. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── **orchestrator.py**     # Updated to handle Step 7
│       ├── domain_models/
│       │   ├── **config.py**       # Added DeltaConfig
│       ├── trainer/
│       │   ├── **delta.py**        # Delta Learning Logic
│       │   └── **ace_trainer.py**  # Refactored to support "load_potential"
└── tests/
    ├── unit/
    │   ├── **test_delta_config.py**
    └── integration/
        └── **test_pipeline_step7.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)

*   **`DeltaConfig`**:
    *   `enable`: bool
    *   `dft_weight`: float (e.g., 100.0) - Weight for DFT structures in loss function.
    *   `surrogate_weight`: float (e.g., 0.1) - Weight for Surrogate structures.
    *   `learning_rate`: float (Usually smaller than base training).
    *   `max_epochs`: int

### 3.2. Delta Trainer (`trainer/delta.py`)

*   **`DeltaTrainer(PacemakerWrapper)`**:
    *   Inherits from the Base Trainer but overrides the configuration generation logic.
    *   **Logic**:
        1.  **Dataset merging**: It creates a combined dataset `step7_mixed_data.pdata` containing both `step2_active_set` (DFT) and `step5_surrogate_labeled` (MACE).
        2.  **Weight Assignment**: It assigns high weights to frames from the active set and low weights to surrogate frames.
        3.  **Initialization**: It sets the `load_potential` parameter in Pacemaker to point to `work_dir/models/ace_base.yace`.
    *   **Goal**: Minimize $L = w_{dft} \sum (E_{ace} - E_{dft})^2 + w_{surr} \sum (E_{ace} - E_{mace})^2$.

### 3.3. Orchestrator Update

*   **`run_step7()`**:
    1.  Check if `step2_active_set.xyz` (DFT) and `ace_base.yace` exist.
    2.  Initialize `DeltaTrainer`.
    3.  Execute training.
    4.  Save final model to `work_dir/final_potential.yace`.

## 4. Implementation Approach

### Step 4.1: Weight Handling
1.  Extend `utils/io.py` to support writing specific frame weights into the Pacemaker data file.
2.  Implement logic in `DeltaTrainer` to iterate over both datasets and assign weights.

### Step 4.2: Pacemaker Configuration
1.  Modify `input.yaml` generation to include `read_from_potential: path/to/base.yace`.
2.  Ensure that the basis set definition is identical to the base model (freeze the basis, only retrain coefficients).

### Step 4.3: Orchestrator Logic
1.  Implement the final step linkage.
2.  Add a validation step: Compare the RMSE of the Base Model vs. the Final Model on the DFT set to quantify improvement.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_delta_config.py`**:
    *   Verify that setting `dft_weight` correctly propagates to the generated dataset file (if weights are stored in data) or input config.
*   **`test_weighting_logic.py`**:
    *   Create two dummy datasets. Merge them. Assert the output file has correct weight columns.

### 5.2. Integration Testing
*   **`test_pipeline_step7.py`**:
    *   **Prerequisite**: A base model and a dummy DFT dataset.
    *   **Step 7**: Run Delta Learning (Mock).
    *   **Verification**:
        *   `final_potential.yace` is created.
        *   The Pacemaker log shows that the initial potential was loaded.
        *   (If possible with Mock) The loss on the DFT set decreases after a few epochs.
