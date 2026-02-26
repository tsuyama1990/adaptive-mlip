# Cycle 03 Specification: MACE Surrogate Loop

## 1. Summary
This cycle implements the "Knowledge Adaptation" and "Massive Sampling" phases of the pipeline, corresponding to **Steps 3 and 4** of the workflow.

First, we implement the `MaceTrainer` to fine-tune the foundation model (MACE-MP-0) using the small but high-value dataset collected and labeled in Cycle 02. This transforms the general-purpose model into a specialized surrogate for the target system.

Second, we implement the `MDSampler`. Armed with the fine-tuned MACE model, we can now run Molecular Dynamics (MD) simulations at near-DFT accuracy but at a fraction of the cost. This allows us to explore the local potential energy surface (PES) exhaustively, generating thousands of diverse configurations that capture thermal fluctuations and rare events.

## 2. System Architecture

The following file structure will be created or modified. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── **orchestrator.py**     # Updated to handle Steps 3 & 4
│       ├── domain_models/
│       │   ├── **config.py**       # Added TrainingConfig & MDConfig
│       ├── trainer/
│       │   ├── **__init__.py**
│       │   └── **mace_trainer.py** # MACE Fine-tuning Logic
│       ├── structure_generator/
│       │   └── **md.py**           # Molecular Dynamics Sampler
└── tests/
    ├── unit/
    │   ├── **test_mace_trainer.py**
    │   └── **test_md_sampler.py**
    └── integration/
        └── **test_pipeline_step3_4.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/config.py`)

*   **`MaceTrainingConfig`**:
    *   `base_model`: str (e.g., "MACE-MP-0")
    *   `epochs`: int (Default: 50)
    *   `batch_size`: int
    *   `learning_rate`: float
    *   `device`: str ("cuda" or "cpu")

*   **`MDConfig`**:
    *   `temperature`: float (Kelvin)
    *   `pressure`: Optional[float] (Bar)
    *   `n_steps`: int
    *   `time_step`: float (fs)
    *   `sampling_interval`: int (Save every N steps)

### 3.2. MACE Trainer (`trainer/mace_trainer.py`)

*   **`MaceFinetuner(ITrainer)`**:
    *   **Responsibilities**:
        1.  Prepare training data (XYZ -> MACE internal format if needed).
        2.  Construct the training command or call the Python API (`mace.tools.train`).
        3.  Monitor training progress (parse logs).
        4.  Save the best model to `work_dir/models/mace_finetuned.model`.
    *   **Constraints**: Must handle GPU availability checks gracefully.

### 3.3. MD Sampler (`structure_generator/md.py`)

*   **`MDSampler(IGenerator)`**:
    *   **Responsibilities**: Run MD using ASE.
    *   **Input**: Starting structures (from Step 2 active set), a Calculator (MaceSurrogate), and `MDConfig`.
    *   **Logic**:
        1.  Initialize velocities (Maxwell-Boltzmann).
        2.  Attach `Langevin` or `NPT` thermostat/barostat.
        3.  Run for `n_steps`.
        4.  Collect frames at `sampling_interval`.
    *   **Output**: A list of `AtomStructure` objects (Unlabeled, representing the trajectory).

### 3.4. Orchestrator Update

*   **`run_step3()`**:
    1.  Load `step2_active_set.xyz`.
    2.  Initialize `MaceFinetuner`.
    3.  Train model.
    4.  Update `GlobalState` with path to new model.

*   **`run_step4()`**:
    1.  Load fine-tuned model into `MaceSurrogate`.
    2.  Load starting structures (subset of `step2_active_set.xyz`).
    3.  Initialize `MDSampler`.
    4.  Run MD.
    5.  Save trajectory to `step4_surrogate_data.xyz`.

## 4. Implementation Approach

### Step 4.1: MACE Trainer
1.  Implement `MaceFinetuner`.
2.  Use `subprocess` to call `mace_run_train` if direct API usage is too complex or unstable.
3.  Implement a "Mock Trainer" mode that simply copies the base model or sleeps for a few seconds (for CI/CD without GPUs).

### Step 4.2: MD Sampler
1.  Implement `MDSampler`.
2.  Use `ase.md.langevin.Langevin` for NVT ensembles.
3.  Ensure the calculator attached is the `MaceSurrogate` (or Mock).

### Step 4.3: Orchestrator Integration
1.  Chain Step 2 -> Step 3 -> Step 4.
2.  Ensure `work_dir/models` directory is created.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_mace_trainer.py`**:
    *   Verify that `MaceFinetuner` generates the correct command-line arguments based on `MaceTrainingConfig`.
    *   Test error handling if the base model file is missing.
*   **`test_md_sampler.py`**:
    *   Use a Mock Calculator (LJ).
    *   Run `MDSampler` for 100 steps.
    *   Assert that the returned trajectory has `100 / interval` frames.
    *   Check that temperature is approximately correct (within fluctuation limits).

### 5.2. Integration Testing
*   **`test_pipeline_step3_4.py`**:
    *   **Prerequisite**: `step2_active_set.xyz` exists.
    *   **Step 3**: Run fine-tuning (Mock mode). Check `mace_finetuned.model` is created.
    *   **Step 4**: Run MD using the fine-tuned model. Check `step4_surrogate_data.xyz` is created and contains valid structures.
