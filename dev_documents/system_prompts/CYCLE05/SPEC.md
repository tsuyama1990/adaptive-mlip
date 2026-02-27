# Cycle 05: Delta Learning (Fine-tuning with DFT)

## 1. Summary
Cycle 05 implements the final and most crucial step of the pipeline: **Delta Learning (Step 7)**. This phase corrects the systematic errors of the base ACE potential (trained on MACE pseudo-labels in Cycle 04) by fine-tuning it against the high-accuracy DFT data collected in the Active Learning phase (Cycle 02).

Instead of training from scratch on the sparse DFT data (which would overfit), or training on a simple mix (where the massive pseudo-data would drown out the DFT signal), we employ a "Delta Learning" strategy. This involves re-weighting the loss function to heavily prioritize the DFT samples while using the base potential as a strong prior or initialization.

## 2. System Architecture

### 2.1. File Structure
**Files to be created/modified in this cycle are bolded.**

```text
src/
└── pyacemaker/
    ├── domain_models/
    │   └── **delta_learning.py**       # Delta Learning Config
    ├── modules/
    │   └── **pacemaker_delta.py**      # Extension of PacemakerTrainer
    ├── orchestrator.py                 # Update with Step 7 logic
    └── utils/
        └── **weighting.py**            # Data Weighting Utilities
```

### 2.2. Component Interaction
1.  **Orchestrator** loads `BaseACEPotential` (from Cycle 04).
2.  **Orchestrator** loads `LabelledActiveSet` (DFT data from Cycle 02).
3.  **`PacemakerDeltaTrainer`** initializes with `DeltaConfig`.
4.  **`PacemakerDeltaTrainer`** constructs a mixed dataset: `PseudoLabelledData` + `LabelledActiveSet`.
5.  **`PacemakerDeltaTrainer`** applies high weights (e.g., 10x-100x) to the `LabelledActiveSet` samples in the loss function.
6.  **`PacemakerDeltaTrainer`** runs a short fine-tuning cycle (e.g., 10-50 epochs).
7.  **`PacemakerDeltaTrainer`** returns the `FinalPotential.yace`.

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/delta_learning.py`)

#### `DeltaConfig`
*   **Fields:**
    *   `base_potential_path`: `Path`
    *   `dft_weight`: `float` (Default: 10.0)
    *   `n_epochs`: `int` (Default: 50)
    *   `l2_regularization`: `float`

### 3.2. Delta Trainer (`modules/pacemaker_delta.py`)
*   **Inherits:** `PacemakerTrainer`
*   **Methods:**
    *   `train_delta(base_dataset: List[AtomStructure], dft_dataset: List[AtomStructure], config: DeltaConfig) -> Path`
*   **Logic:**
    *   Combine datasets.
    *   Assign sample weights: 1.0 for base, `dft_weight` for DFT.
    *   Set `initial_potential` in `input.yaml` to `base_potential_path`.
    *   Run `pacemaker` with `finetune_mode=True`.

## 4. Implementation Approach

1.  **Config:** Implement `DeltaConfig`.
2.  **Trainer:** Extend `PacemakerTrainer` to support `train_delta`. Key addition is handling per-structure weights in the input data format (e.g., adding a `weight` column in `data.p4`).
3.  **Orchestrator Update:** Add `run_step7()`.
4.  **Weighting:** Implement helper functions to merge datasets and assign weights.

## 5. Test Strategy

### 5.1. Unit Tests
*   **Weight Assignment:**
    *   Create a mixed dataset.
    *   Verify that DFT samples have the correct weight property.
*   **Config Generation:**
    *   Verify `input.yaml` contains `load_potential: base.yace` (or equivalent).

### 5.2. Integration Tests
*   **Delta Learning Flow:**
    *   Mock the training.
    *   Verify that the trainer receives both datasets.
    *   Check that the output potential is different from the input potential (file hash/timestamp).
