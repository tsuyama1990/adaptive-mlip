# Cycle 04: Distillation Phase (Pacemaker Base Training)

## 1. Summary
Cycle 04 focuses on "Distilling" the knowledge of the fine-tuned MACE model into a lightweight Polynomial (ACE) potential. The process involves two key steps:
1.  **Surrogate Labeling (Step 5):** The MACE model, which has been fine-tuned on the small active set, acts as a high-fidelity "oracle" to label the large surrogate pool generated in Cycle 03. This creates a massive "pseudo-labelled" dataset (e.g., 10,000 structures) at zero DFT cost.
2.  **Pacemaker Base Training (Step 6):** We use this pseudo-labelled dataset to train the base ACE potential using `pacemaker`. This model learns the general potential energy surface (PES) topology from MACE.

## 2. System Architecture

### 2.1. File Structure
**Files to be created/modified in this cycle are bolded.**

```text
src/
└── pyacemaker/
    ├── domain_models/
    │   └── **pacemaker.py**            # Pacemaker Config & Paths
    ├── modules/
    │   └── **pacemaker_trainer.py**    # Pacemaker Wrapper
    ├── orchestrator.py                 # Update with Step 5 & 6 logic
    └── utils/
        └── **yaml.py**                 # Safe YAML Writer for Pacemaker input
```

### 2.2. Component Interaction
1.  **Orchestrator** loads `SurrogatePool` (from Cycle 03).
2.  **`MaceOracle`** (reused from Cycle 02) labels the pool with energy, forces, and stress.
3.  **Orchestrator** saves the result as `PseudoLabelledData` (XYZ/CFG).
4.  **`PacemakerTrainer`** initializes with `PacemakerConfig` (cutoff, order, etc.).
5.  **`PacemakerTrainer`** generates `input.yaml` for `pacemaker`.
6.  **`PacemakerTrainer`** runs `pacemaker` (subprocess).
7.  **`PacemakerTrainer`** returns the path to `potential.yace`.

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/pacemaker.py`)

#### `PacemakerConfig`
*   **Fields:**
    *   `cutoff`: `float` (Default: 5.0)
    *   `order`: `int` (Default: 3)
    *   `elements`: `List[str]`
    *   `b_basis`: `str` (Default: "mace") # Basis set type
    *   `loss_energy`: `float`
    *   `loss_forces`: `float`
    *   `loss_stress`: `float`
    *   `weight_dft`: `float` (Relevant for Step 7, but defined here)

### 3.2. Pacemaker Trainer (`modules/pacemaker_trainer.py`)
*   **Interface:** `BaseTrainer`
*   **Methods:**
    *   `train(dataset: List[AtomStructure], config: PacemakerConfig) -> Path`
*   **Logic:**
    *   Convert `AtomStructure` list to `data.p4` or extended XYZ format expected by `pacemaker`.
    *   Generate `input.yaml` with the correct fitting settings.
    *   Execute `pacemaker input.yaml`.
    *   Parse the output log to monitor convergence.
    *   Return path to the best potential (`output_potential.yace`).

## 4. Implementation Approach

1.  **Pacemaker Config:** Implement `PacemakerConfig` model.
2.  **Trainer:** Implement `PacemakerTrainer` wrapper. Ensure it handles the environment variables (e.g., `OMP_NUM_THREADS`) correctly.
3.  **Orchestrator Update:** Add `run_step5()` (Labeling) and `run_step6()` (Training).
4.  **YAML Utility:** Implement `utils.yaml` to ensure clean YAML generation for Pacemaker.

## 5. Test Strategy

### 5.1. Unit Tests
*   **Config Generation:**
    *   Create a `PacemakerConfig`.
    *   Generate the YAML string.
    *   Assert that keys like `cutoff` and `elements` are correctly formatted.

### 5.2. Integration Tests
*   **Labeling:**
    *   Run `MaceOracle.compute_batch` on 10 structures.
    *   Verify all have energy/forces.
*   **Training Loop:**
    *   Mock the `pacemaker` subprocess call (or run a tiny fit if fast enough).
    *   Verify that `input.yaml` is created on disk.
    *   Verify that the trainer returns a path ending in `.yace`.
