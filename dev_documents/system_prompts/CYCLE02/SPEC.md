# Cycle 02: Smart Sampling & Active Learning

## 1. Summary
Cycle 02 implements the critical first two steps of the MACE Knowledge Distillation pipeline: **DIRECT Sampling** and **Active Learning**. This cycle focuses on generating a diverse initial pool of atomic structures that maximally covers the configuration space, and then intelligently selecting the most informative subset for expensive DFT calculations.

By moving away from random sampling, we aim to reduce the data requirements by an order of magnitude. The `DirectSampler` maximizes entropy in descriptor space, ensuring no two structures are too similar. The `MaceOracle` is enhanced to compute uncertainty (variance) from an ensemble of MACE models (or a single model with MC Dropout), allowing us to pinpoint configurations where the current knowledge is weakest.

## 2. System Architecture

### 2.1. File Structure
**Files to be created/modified in this cycle are bolded.**

```text
src/
└── pyacemaker/
    ├── domain_models/
    │   ├── **active_learning.py**      # Active Learning Config & Results
    │   └── data.py                     # Enhanced with uncertainty fields
    ├── modules/
    │   ├── **sampling.py**             # DIRECT Sampling Implementation
    │   └── **mace_oracle.py**          # MACE Wrapper with Uncertainty
    ├── orchestrator.py                 # Update with Step 1 & 2 logic
    └── utils/
        └── **descriptors.py**          # ACE/SOAP Descriptor Calculation
```

### 2.2. Component Interaction
1.  **Orchestrator** calls `DirectSampler.generate(n=100)`.
2.  **`DirectSampler`** generates random structures, computes descriptors (ACE/SOAP), and uses a greedy farthest-point sampling strategy to maximize diversity. Returns `InitialPool`.
3.  **Orchestrator** passes `InitialPool` to `MaceOracle.compute_uncertainty()`.
4.  **`MaceOracle`** loads a pre-trained MACE model (e.g., MACE-MP-0), runs inference, and attaches `uncertainty` scores to each structure.
5.  **Orchestrator** filters the pool (e.g., top 10% uncertainty) to create the `ActiveSet`.
6.  **Orchestrator** passes `ActiveSet` to `DftOracle` (or Mock) for ground-truth labelling.

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/active_learning.py`)

#### `ActiveLearningConfig`
*   **Fields:**
    *   `method`: `str` (e.g., "entropy", "uncertainty")
    *   `n_initial`: `int` (Number of structures to generate)
    *   `n_active`: `int` (Number of structures to select for DFT)
    *   `descriptor`: `DescriptorConfig` (ACE/SOAP settings)

#### `SamplingResult`
*   **Fields:**
    *   `pool`: `List[AtomStructure]`
    *   `descriptors`: `np.ndarray` (N x D matrix)
    *   `selection_indices`: `List[int]`

### 3.2. DIRECT Sampling (`modules/sampling.py`)
Implementation of the "Entropy Maximization" strategy.
*   **Algorithm:**
    1.  Generate a large candidate pool (N=1000) using random distortions (`RattledBulk`, `Surface`, `Cluster`).
    2.  Compute global descriptors for all candidates.
    3.  Select the first point randomly.
    4.  Select subsequent points that maximize the minimum distance to the already selected set (MaxMin diversity).
    5.  Return the top `n_initial` structures.

### 3.3. MACE Uncertainty (`modules/mace_oracle.py`)
*   **Interface:** `BaseOracle`
*   **Methods:**
    *   `compute_uncertainty(structures: List[AtomStructure]) -> List[AtomStructure]`
*   **Logic:**
    *   If using an ensemble: Run inference on all models, compute variance of energy/forces.
    *   If using a single model: Use the built-in uncertainty output if available, or last-layer variance.
    *   Store the result in `AtomStructure.uncertainty`.

## 4. Implementation Approach

1.  **Descriptor Utility:** Implement `utils.descriptors.py` using `mace` or `dscribe` to compute global structure vectors.
2.  **Sampler:** Implement `DirectSampler` in `modules/sampling.py`. Focus on the MaxMin selection logic.
3.  **MACE Wrapper:** Implement `MaceOracle` in `modules/mace_oracle.py`. Ensure it can load a model from a path or URI (e.g., `MACE-MP-0`).
4.  **Orchestrator Update:** Add `run_step1()` and `run_step2()` methods to the `Orchestrator` class.
5.  **Config Update:** Add `ActiveLearningConfig` to `PyAceConfig`.

## 5. Test Strategy

### 5.1. Unit Tests
*   **Sampling Diversity:**
    *   Generate 100 structures.
    *   Compute pairwise distances.
    *   Assert that the minimum distance is significantly higher than random sampling.
*   **MACE Loading:**
    *   Test loading a small dummy MACE model.
    *   Verify `compute_uncertainty` returns non-negative values.

### 5.2. Integration Tests
*   **Step 1-2 Flow:**
    *   Run `Orchestrator.run_step1()` -> Check `initial_pool.xyz` exists.
    *   Run `Orchestrator.run_step2()` -> Check `active_set.xyz` exists and has `uncertainty` property.
    *   Verify that `active_set` size equals `n_active`.
