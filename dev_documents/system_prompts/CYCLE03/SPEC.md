# Cycle 03: Surrogate Refinement (MACE Fine-tuning)

## 1. Summary
Cycle 03 focuses on transforming the generic MACE foundation model (MACE-MP-0) into a system-specific surrogate. By fine-tuning the model on the small, high-quality `ActiveSet` generated in Cycle 02 (labelled with DFT), we create a "smart" generator capable of running stable Molecular Dynamics (MD) simulations. This fine-tuned model then generates thousands of diverse structures (Step 4) that serve as the training data for the final ACE potential.

The key challenge here is to ensure the fine-tuning process is robust even with sparse data (e.g., < 50 structures) and that the subsequent MD simulations do not explode due to unphysical forces.

## 2. System Architecture

### 2.1. File Structure
**Files to be created/modified in this cycle are bolded.**

```text
src/
└── pyacemaker/
    ├── domain_models/
    │   └── **training.py**             # Training Config & Paths
    ├── modules/
    │   ├── **mace_trainer.py**         # MACE Fine-tuning Wrapper
    │   └── **md_generator.py**         # MD Simulation Engine
    ├── orchestrator.py                 # Update with Step 3 & 4 logic
    └── utils/
        └── **md.py**                   # ASE MD Helpers (Thermostats, Logging)
```

### 2.2. Component Interaction
1.  **Orchestrator** loads `ActiveSet` (labelled with DFT).
2.  **`MaceTrainer`** initializes with `MaceConfig` (base model path, learning rate, epochs).
3.  **`MaceTrainer`** runs the fine-tuning loop (calling `mace.cli.run_train` internally or via API).
4.  **`MaceTrainer`** saves the `finetuned_mace.model`.
5.  **Orchestrator** passes `finetuned_mace.model` to `MdGenerator`.
6.  **`MdGenerator`** runs multiple parallel MD trajectories (NVT/NPT) starting from the `ActiveSet` structures.
7.  **`MdGenerator`** samples frames at regular intervals to create the `SurrogatePool` (e.g., 1000 structures).

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/training.py`)

#### `MaceTrainingConfig`
*   **Fields:**
    *   `base_model`: `str` (Path or "MACE-MP-0")
    *   `epochs`: `int` (Default: 50)
    *   `batch_size`: `int` (Default: 4)
    *   `learning_rate`: `float` (Default: 0.01)
    *   `loss_energy`: `float` (Weight for energy loss)
    *   `loss_forces`: `float` (Weight for force loss)

#### `MdConfig`
*   **Fields:**
    *   `temperature`: `float` (K)
    *   `pressure`: `float` (bar)
    *   `steps`: `int`
    *   `timestep`: `float` (fs)
    *   `sampling_interval`: `int`

### 3.2. MACE Trainer (`modules/mace_trainer.py`)
*   **Interface:** `BaseTrainer`
*   **Methods:**
    *   `train(dataset: List[AtomStructure], config: MaceTrainingConfig) -> Path`
*   **Logic:**
    *   Convert `AtomStructure` list to MACE-compatible XYZ format.
    *   Construct the MACE command line arguments or Python API call.
    *   Execute training.
    *   Return path to the best checkpoint.

### 3.3. MD Generator (`modules/md_generator.py`)
*   **Interface:** `BaseGenerator`
*   **Methods:**
    *   `generate(n_structures: int, initial_structures: List[AtomStructure], model_path: Path) -> List[AtomStructure]`
*   **Logic:**
    *   Load the fine-tuned model as an ASE Calculator.
    *   For each initial structure, run an NVT/NPT simulation.
    *   Check for "unphysical" events (e.g., atoms getting too close, energy spikes) and terminate/prune if detected.
    *   Collect frames to form the `SurrogatePool`.

## 4. Implementation Approach

1.  **MACE Trainer:** Implement `MaceTrainer` to wrap the `mace` library. Focus on handling the `foundations` model loading correctly.
2.  **MD Engine:** Implement `MdGenerator` using `ase.md.langevin` or `ase.md.npt`. Add safeguards for exploding simulations.
3.  **Orchestrator Update:** Add `run_step3()` and `run_step4()`.
4.  **Config Update:** Add `MaceTrainingConfig` and `MdConfig`.

## 5. Test Strategy

### 5.1. Unit Tests
*   **Trainer Configuration:**
    *   Verify that `MaceTrainingConfig` correctly formats CLI arguments.
*   **MD Stability:**
    *   Run a short MD (10 steps) using `MockOracle` as the calculator.
    *   Verify that temperature fluctuates around the target.

### 5.2. Integration Tests
*   **Fine-tuning Flow:**
    *   Mock the actual training (since it requires GPU/Time).
    *   Verify that `MaceTrainer` produces a model file artifact.
*   **Surrogate Generation:**
    *   Load the mocked model.
    *   Run `MdGenerator`.
    *   Verify `SurrogatePool` contains `n_structures` valid atoms objects.
