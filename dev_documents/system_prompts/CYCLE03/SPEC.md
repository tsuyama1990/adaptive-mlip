# CYCLE 03: Hierarchical Distillation Loop (Phase 1 & 4)

## 1. Summary

This cycle integrates the core active learning components into the new 4-phase architecture: Zero-Shot Distillation (Phase 1) and Hierarchical Fine-Tuning (Phase 4). We will introduce the `MACEManager` and the `TieredOracle` to `core.oracle`, allowing the system to use a foundational model for fast, high-quality structure filtering before ever calling expensive DFT (Quantum Espresso). Furthermore, we will upgrade the `PacemakerTrainer` in `core.trainer` to support true incremental (Delta) learning, mitigating catastrophic forgetting by utilizing a fixed-size Replay Buffer of past configurations.

## 2. System Architecture

This cycle focuses on the Oracle (evaluator) and Trainer (updater) domains.

```text
src/pyacemaker/
├── core/
│   ├── base.py                 (No changes)
│   ├── engine.py               (No changes from Cycle 02)
│   ├── **oracle.py**           (Modify: Add MACEManager, TieredOracle)
│   ├── **trainer.py**          (Modify: Incremental update & Replay Buffer)
│   └── loop.py                 (No changes)
├── domain_models/
│   ├── config.py               (No changes)
│   └── data.py                 (No changes)
└── interfaces/
    └── qe_driver.py            (No changes)
```

## 3. Design Architecture

The primary goal is to minimize DFT calls while maximizing the information content of the training set.

### 3.1. Oracle Module (`core/oracle.py`)

We will introduce two new classes implementing the `BaseOracle` interface.

**`MACEManager(BaseOracle)`:**
*   A wrapper around the `mace` Python package (or CLI, depending on integration preference; Python API is strongly preferred for speed).
*   **Key Responsibilities:** Takes an ASE `Atoms` object, runs MACE inference, and populates `atoms.calc.results` with `energy`, `forces`, and critically, `uncertainty` (e.g., ensemble variance or distance in latent space).
*   **Invariants:** Must never fail on unphysical structures; it should return high uncertainty instead of crashing.

**`TieredOracle(BaseOracle)`:**
*   A routing Oracle. It contains instances of both `MACEManager` and `QEDriver` (DFT).
*   **Routing Logic:** When asked to evaluate a structure, it first calls `MACEManager`.
    *   If `mace_uncertainty <= DistillationConfig.uncertainty_threshold` (Phase 1 logic), it returns the MACE result immediately.
    *   If `mace_uncertainty > ActiveLearningThresholds.threshold_call_dft` (Phase 3 logic), it routes the structure to `QEDriver` for ground-truth evaluation.
*   **Invariants:** The caller (the Orchestrator) should not know or care which underlying engine performed the calculation. The returned `Atoms` object must simply contain valid `energy` and `forces`.

### 3.2. Trainer Module (`core/trainer.py`)

We will upgrade the `PacemakerTrainer` (or create an `IncrementalTrainer` subclass) to support Phase 4 requirements.

**`IncrementalTrainer` (or modified `PacemakerTrainer`):**
*   **Replay Buffer Integration:** Instead of taking a single list of `new_structures` and retraining from scratch, the `train()` method now manages a persistent on-disk buffer (e.g., `training_history.extxyz`).
*   **Delta Learning Strategy:** When `train(new_structures, config: LoopStrategyConfig)` is called:
    1.  Append `new_structures` to the master history file.
    2.  If the history exceeds `replay_buffer_size`, randomly sample (or use D-Optimality/MaxVol if feasible, though random uniform is safer for basic stability) `replay_buffer_size` structures.
    3.  Generate the `input.yaml` for Pacemaker. Crucially, set the `initial_potential` field to the path of the *previous* iteration's `.yace` file, rather than starting from random weights.
    4.  Configure the `fit` section to optimize only the difference (Delta) between the current weights and the target data, drastically reducing epochs required.
*   **MACE Finetune Wrapper (Optional but highly recommended):** A lightweight `FinetuneManager` that calls MACE's finetuning script on the `new_structures` (the exact DFT calculations of the extracted clusters) before generating the massive surrogate dataset for Pacemaker.

## 4. Implementation Approach

1.  **Implement `MACEManager`:**
    *   Create the class in `core/oracle.py`.
    *   Implement the `evaluate(atoms)` method. Initialize the `mace.calculators.mace_mp` calculator (or equivalent) once during `__init__` to avoid reloading weights on every call.
    *   Extract forces, energy, and uncertainty from the calculator results and attach them to the returned `Atoms` object.
2.  **Implement `TieredOracle`:**
    *   Create the class in `core/oracle.py`, taking instances of a fast oracle (MACE) and slow oracle (QE) in its constructor.
    *   Implement `evaluate(atoms)`. Add logic to check `atoms.info.get('uncertainty', 0.0)` against the configured thresholds to decide routing.
3.  **Upgrade `PacemakerTrainer` for Incremental Updates:**
    *   Modify the `train()` method signature or internal state to handle history.
    *   Implement the Replay Buffer logic: Read existing `history.extxyz`, append `new_structures`, shuffle/sample to `replay_buffer_size`, and write out a temporary `training_set.extxyz`.
    *   Modify the `input.yaml` generator to conditionally include `initial_potential: current.yace` if `incremental_update` is True in `LoopStrategyConfig`.
4.  **Refine & Lint:** Run Ruff and MyPy. Ensure strict typing for all Oracle inputs/outputs.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)

Unit tests will focus on the routing logic of the `TieredOracle` and the YAML generation/buffer management of the `IncrementalTrainer`.

1.  **`TieredOracle` Routing:**
    *   Mock both a fast Oracle and a slow Oracle.
    *   Configure the fast Oracle to return an `Atoms` object with `info['uncertainty'] = 0.01`.
    *   Call `TieredOracle.evaluate()`.
    *   **Assertions:** The fast Oracle's evaluate method must be called exactly once. The slow Oracle's evaluate method must NOT be called. The returned `Atoms` object must be the one from the fast Oracle.
    *   Change the fast Oracle mock to return `info['uncertainty'] = 0.1` (above threshold).
    *   Call `TieredOracle.evaluate()`.
    *   **Assertions:** The fast Oracle is called once. The slow Oracle is called once. The returned `Atoms` object must be the one from the slow Oracle.
2.  **`IncrementalTrainer` Replay Buffer:**
    *   Create a mock history file containing 10 dummy `Atoms` objects.
    *   Configure `replay_buffer_size = 15`. Pass 10 new `Atoms` objects to `train()`.
    *   **Assertions:** The resulting training set must contain exactly 15 structures (a mix of history and new data). The master history file must now contain 20 structures.
3.  **Delta Learning YAML Generation:**
    *   Call the `train()` method with `incremental_update=True` and a valid previous potential path.
    *   **Assertions:** Parse the generated `input.yaml` (using `yaml.safe_load`). Verify the `initial_potential` key exists and points to the correct path. Verify `replay_buffer_size` constraints were respected in the data section.

### Integration Testing Approach (Min 300 words)

Integration testing will simulate Phase 1 (Zero-Shot Distillation) to ensure the `MACEManager` and `IncrementalTrainer` interact correctly without DFT.

1.  **Zero-Shot Distillation Pipeline (Mock MACE):**
    *   Since loading real MACE weights might be slow/unavailable in CI, use a dummy calculator that returns random energies and random uncertainties between 0.0 and 0.1.
    *   Generate a large pool of 1000 random structures (e.g., using `RandomRattlePolicy`).
    *   Pass the pool through `MACEManager` (mocked).
    *   Filter the results using `DistillationConfig.uncertainty_threshold = 0.05`.
    *   Pass the filtered structures (approx. 500) to `IncrementalTrainer`.
    *   **Assertions:**
        *   The filtering step correctly removes all structures with uncertainty > 0.05.
        *   The Trainer successfully generates a training set and an `input.yaml`.
        *   (Optional if Pacemaker is installed in the test environment): Run a 1-epoch Pacemaker training job to verify the generated config and `.extxyz` files are syntactically valid and mathematically sound for ACE fitting.
2.  **Phase 4 (Surrogate Generation & Delta Update) Pipeline:**
    *   Simulate an extraction event: create one "clean" DFT-evaluated cluster.
    *   Mock a MACE Finetune step (simply pass the cluster through).
    *   Use the mocked `MACEManager` to generate 50 small random perturbations of the cluster (Surrogate Generation).
    *   Pass the surrogates + the 1 DFT structure to the `IncrementalTrainer`.
    *   **Assertions:** The Replay Buffer correctly mixes these new 51 structures with historical data, and the generated Pacemaker config correctly sets up a Delta Learning run starting from the base potential.
