# SYSTEM ARCHITECTURE
**Status**: DRAFT
**Version**: 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date**: 2026-02-28

## 1. Summary
The PYACEMAKER version 2.1.0 introduces a paradigm shift in constructing Machine Learning Interatomic Potentials (MLIPs). Building upon the foundations established in Phase 01, this architecture targets High-Performance Computing (HPC) environments for long-timescale, large-scale Molecular Dynamics (MD) simulations. It addresses the critical physical and system constraints found in the previous iteration—such as MD time-continuity breaks, false positive halts due to thermal noise, physical divergence during local cluster extraction, and catastrophic forgetting caused by slow batch retraining.

By integrating the "Master-Slave Inversion" and "Hierarchical Distillation" concepts derived from FLARE best practices, this architecture enables the system to safely pause an MD simulation, extract and physically repair problematic local clusters, run efficient Quantum Espresso (QE) DFT calculations, incrementally fine-tune the MACE foundation model and the target ACE potential, and seamlessly resume the simulation without resetting time.

The architecture is purely additive. It reuses the core configuration structures and exploration policies from Phase 01 while extending them with robust, modern design patterns such as Dependency Injection, Repository Pattern, and tiered execution strategies.

## 2. System Design Objectives
The primary design objectives for the PYACEMAKER NextGen architecture are focused on bridging the gap between small-scale theoretical model generation and practical, large-scale HPC deployments.

First and foremost, the system must guarantee MD Time Continuity. In the previous iteration, anytime the simulation encountered high uncertainty, the loop halted and restarted from scratch, making it impossible to study long-term diffusion or phase transformation phenomena. The new design mandates a "Master-Slave Resume" capability. LAMMPS must hold the state in C++ or via secure restart files while the Python orchestrator updates the underlying MLIP, allowing the simulation to proceed seamlessly from the exact microsecond and coordinate state where it paused.

Secondly, the system must establish robust Thermal Noise Resistance. Instead of a single uncertainty threshold that triggers expensive DFT calculations on harmless thermal fluctuations, a "Two-Tier Evaluator" is implemented. This involves `threshold_call_dft` to pause the simulation only when the system uncertainty is consistently high over several steps, and `threshold_add_train` to pinpoint the specific atoms (the "epicentre") that require retraining.

Thirdly, Intelligent Cluster Extraction is critical to prevent physical divergence. Previously, cutting out local clusters for DFT created massive amounts of dangling bonds, causing charge imbalances and dipole moment divergence, which either crashed the DFT SCF loop or resulted in "garbage" data. The system must now employ safe spherical extraction (`force_weight=1.0` for core, `0.0` for buffer), Pre-Relaxation of the buffer zone using the MACE foundation model while keeping the core frozen, and Auto-Passivation (e.g., adding dummy Hydrogen atoms) to ensure the cluster is physically and electronically neutral before it is passed to Quantum Espresso.

Finally, the architecture demands O(1) Computational Cost for Training. The legacy system suffered from O(N) cost scaling and catastrophic forgetting because it retrained the ACE potential from scratch using all historical data. The new requirement introduces Hierarchical Delta Learning: fine-tuning the MACE model on the specific DFT data, generating thousands of surrogate data points instantaneously via MACE, and updating the ACE potential incrementally (Delta Learning) while mixing in a fixed-size Replay Buffer of past data. This ensures the system retains its generalized knowledge of the bulk structure while quickly adapting to the new defect or interfacial physics encountered during the MD run.

Success criteria include the ability to run 10-million atom MD simulations continuously with intelligent pauses, maintaining a >99% DFT SCF convergence rate on extracted clusters, and executing the active learning loop in a strictly bounded memory footprint via iterators and streams.

## 3. System Architecture
The overall architecture transitions from a linear script runner to a robust state machine orchestrated via an event-driven `Orchestrator`. Strict separation of concerns is enforced: `Domain Models` dictate pure data schemas, `Core Engines` manage external execution, and `Interfaces` handle the I/O translation.

**Boundary Management Rules:**
1. **Dependency Injection**: Core components (e.g., `LammpsEngine`, `LammpsScriptGenerator`, `LammpsResultParser`) must strictly enforce dependency injection via their constructors. They must require valid configuration objects without providing `| None = None` fallbacks to ensure type safety.
2. **Streaming Execution**: Massive datasets must never be loaded entirely into memory. Components must use iterators (`ase.io.iread`) and bounded buffers (e.g., `collections.deque`) to prevent OOM errors.
3. **Immutability of Phase 01**: Existing domain models from Phase 01 (like `MDConfig`) will not be modified. New parameters are placed in new models (`DistillationConfig`, `ActiveLearningThresholds`) to prevent breaking existing serialization logic.

### 3.1 Architectural Diagram

```mermaid
flowchart TD
    A[Initial State] --> Phase1

    subgraph Phase1 [Phase 1: Zero-Shot Distillation]
    B1[Generate Combinatorial Structures] --> B2[DIRECT Sampling]
    B2 --> B3[MACE Confidence Filtering]
    B3 --> B4[Pacemaker Baseline Train (LJ Delta)]
    end

    Phase1 --> Phase2

    subgraph Phase2 [Phase 2: Validation]
    C1[EOS & Phonon Calc] --> C2{Stable?}
    C2 -- No --> B1
    C2 -- Yes --> C3[Miniature MD Stress Test]
    end

    Phase2 --> Phase3

    subgraph Phase3 [Phase 3: Intelligent Cutout]
    D1[LAMMPS MD Simulation] --> D2{Max Gamma > threshold_call_dft?}
    D2 -- No --> D1
    D2 -- Yes --> D3[Identify Epicentre Atoms > threshold_add_train]
    D3 --> D4[Spherical Cutout & Weighting]
    D4 --> D5[MACE Pre-Relaxation Buffer]
    D5 --> D6[Auto-Passivation]
    D6 --> D7[Clean DFT Calc (QE)]
    end

    Phase3 --> Phase4

    subgraph Phase4 [Phase 4: Hierarchical Fine-Tuning]
    E1[Finetune MACE with DFT Data] --> E2[Generate Surrogate Data via Awakened MACE]
    E2 --> E3[Incremental ACE Train + Replay Buffer]
    E3 --> E4[Master-Slave Resume LAMMPS MD]
    end

    E4 --> D1
```

## 4. Design Architecture
The file structure and component design reflect a modern, scalable approach. Existing structures are maintained, while new components are placed into clearly separated domains.

### 4.1 File Structure Overview
```text
src/pyacemaker/
├── core/
│   ├── engine.py           # Extended LAMMPS handling (Master-Slave inversion)
│   ├── oracle.py           # Extended TieredOracle and MACEManager
│   ├── trainer.py          # FinetuneManager for MACE, IncrementalTrainer for Pacemaker
│   └── policy.py           # Exploration policies (Phase 1 generator)
├── domain_models/
│   ├── config.py           # Phase 01 configurations
│   ├── distillation.py     # NEW: DistillationConfig
│   └── workflow.py         # NEW: ActiveLearningThresholds, CutoutConfig, LoopStrategyConfig
├── interfaces/
│   ├── qe_driver.py        # Existing QE interface (reused)
│   └── pacemaker_wrapper.py# Reused for training
├── utils/
│   ├── extraction.py       # NEW/EXTENDED: extract_intelligent_cluster
│   └── structure.py        # Helpers for passivation
├── orchestrator.py         # Updated to handle the 4-phase state machine
└── factory.py              # DI Container logic
```

### 4.2 Core Domain Pydantic Models Structure
To ensure safety and integration, the new schemas in `domain_models/workflow.py` and `distillation.py` extend the workflow without touching `MDConfig` or `TrainingConfig`.

*   **`DistillationConfig`**: Defines `mace_model_path`, `uncertainty_threshold`, and `sampling_structures_per_system`. Used strictly by Phase 1.
*   **`ActiveLearningThresholds`**: Manages the two-tier thresholds `threshold_call_dft` and `threshold_add_train`, along with `smooth_steps` to handle thermal noise.
*   **`CutoutConfig`**: Specifies `core_radius`, `buffer_radius`, `enable_pre_relaxation`, and `enable_passivation` for the Intelligent Cluster Extraction.
*   **`LoopStrategyConfig`**: Combines `use_tiered_oracle`, `incremental_update`, `replay_buffer_size`, and holds the `thresholds`.

**Integration Points**:
The `PyAceConfig` object will be extended with optional properties mapped to these new Pydantic schemas. Existing serialization functions will continue to work, parsing these models if they are present in the `config.yaml`. The `Orchestrator` will conditionally execute the 4-Phase MACE workflow if `DistillationConfig` is defined; otherwise, it defaults to the legacy Phase 01 loop.

## 5. Implementation Plan

### Cycle 01: Core Extraction & Two-Tier Evaluator
**Objective:** Implement the fundamental logical building blocks required for intelligent processing.
*   **Tasks**:
    *   Create the `DistillationConfig`, `ActiveLearningThresholds`, `CutoutConfig`, and `LoopStrategyConfig` schemas in the domain models.
    *   Redesign `pyacemaker.utils.extraction.py` to implement `extract_intelligent_cluster`.
    *   Implement neighbor-list-based spherical extraction with `force_weight` array assignment.
    *   Implement the MACE-based `_pre_relax_buffer` (freezing core atoms) and `_passivate_surface` (adding H atoms to dangling bonds).
    *   Implement the two-tier threshold evaluation logic to distinguish between halting MD and selecting training epicentres.

### Cycle 02: Master-Slave Inversion & Seamless Resume
**Objective:** Alter the MD execution flow to allow pausing and resuming without resetting the simulation state.
*   **Tasks**:
    *   Update `pyacemaker.core.engine.py` to implement Master-Slave inversion.
    *   Implement the `read_restart` fallback or `fix python/invoke` logic in the `LammpsScriptGenerator` to pause LAMMPS.
    *   Implement the Soft Start protocol: generating LAMMPS scripts that inject a short-damping Langevin thermostat immediately after a restart to prevent energy discontinuities.
    *   Integrate Lennard-Jones (LJ) baseline Delta Learning generation into the Pacemaker training script builder.

### Cycle 03: Hierarchical Distillation Loop Integration
**Objective:** Connect the modules and implement the 4-phase state machine in the Orchestrator.
*   **Tasks**:
    *   Implement `MACEManager` and `TieredOracle` in `pyacemaker.core.oracle.py` to handle the routing of structural evaluations.
    *   Implement `FinetuneManager` for MACE read-out layer updating.
    *   Implement `IncrementalTrainer` logic to manage the fixed-size Replay Buffer using `ase.io.iread` and perform Delta Learning.
    *   Refactor `orchestrator.py` to execute Phase 1 through Phase 4 sequentially, calling the new Managers and evaluating the LoopState correctly.

### Cycle 04: Scale, Robustness, and Validation
**Objective:** Ensure the system scales in HPC environments and passes all non-functional requirements.
*   **Tasks**:
    *   Implement task-level SQLite/JSON checkpointing within `StateManager` to save state at the granularity of individual DFT or surrogate generation steps.
    *   Develop the Artifact Cleanup daemon to automatically gzip or delete massive `.wfc` and LAMMPS dump files immediately after successful parsing.
    *   Implement asynchronous dispatch mechanisms for the `TieredOracle` using `concurrent.futures`.
    *   Finalize UAT documentation and ensure all scenarios run smoothly using the mock strategies.

## 6. Test Strategy

### Cycle 01 Test Strategy
*   **Unit Tests**: Validate `extract_intelligent_cluster` by feeding it a mock massive `ase.Atoms` object and ensuring the returned structure has the exact correct `force_weight` arrays. Test passivation on a simple MgO slab to ensure H atoms are added to exposed Mg/O bonds.
*   **Side-effect Mitigation**: All MACE pre-relaxation calls in `extraction.py` will be intercepted using `unittest.mock.patch` to return mathematically manipulated coordinates without loading PyTorch or the actual MACE `.model` file into memory.

### Cycle 02 Test Strategy
*   **Unit Tests**: Verify the output of `LammpsScriptGenerator` explicitly asserts the presence of `read_restart`, `write_restart`, and `fix langevin` strings when resuming.
*   **Integration Tests**: Run the `LammpsEngine` with a small, 100-atom mock potential, trigger a halt, and verify that the extracted state can be successfully passed back to a new `LammpsEngine` instance, checking that atomic velocities are perfectly preserved.
*   **Side-effect Mitigation**: LAMMPS runs will be executed in `tempfile.TemporaryDirectory()`. The engine will be tested against mock configuration files rather than hardcoding system paths.

### Cycle 03 Test Strategy
*   **Integration Tests**: Test the full `TieredOracle` routing. Pass structures with artificially assigned high and low uncertainties to ensure the low uncertainty ones hit `MACEManager` (mocked) and the high uncertainty ones are routed to `QEDriver` (mocked).
*   **Unit Tests**: Assert that the `IncrementalTrainer` correctly truncates the history file to `replay_buffer_size` and leverages `append=True` during I/O.
*   **Side-effect Mitigation**: All large dataset streaming will be tested using `itertools.islice` on dynamically generated dummy `.extxyz` generators.

### Cycle 04 Test Strategy
*   **E2E Tests**: Run the complete 4-Phase UAT scenario in a "Mock Mode". `MACEManager` and `QEDriver` will be entirely bypassed by returning `0.0` arrays for forces and random floats below thresholds for energies.
*   **Unit Tests**: Intentionally raise exceptions during the inner loops of the Orchestrator and assert that the SQLite/JSON `StateManager` successfully commits the `LoopState` right before the crash.
*   **Side-effect Mitigation**: Background daemon processes for file cleanup will be mocked to verify they are called with the correct file paths without actually triggering OS-level deletions during the test suite execution.