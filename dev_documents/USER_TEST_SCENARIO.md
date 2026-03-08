# PYACEMAKER v2.1.0 User Acceptance Testing (UAT) and Tutorial Scenarios

This document outlines the user-level test scenarios designed from the perspective of a researcher. It verifies that the paradigm shifts introduced in the NextGen Hierarchical Distillation Architecture (Phase 1 through Phase 4) function correctly, resolving the physical and systemic bottlenecks found in long-term HPC Molecular Dynamics (MD) simulations.

## 1. Test Scenarios

### Scenario ID: UAT-PHASE1-001 (Priority: High)
**Title:** Phase 1 - Zero-Shot Distillation and Baseline Construction Verification
**Purpose:** To confirm that a physically valid initial potential (applying Lennard-Jones Delta Learning) is constructed solely through MACE inference, without ever invoking Density Functional Theory (DFT) calculations.
**Prerequisites:**
1.  The input elements are specified as a quaternary system (e.g., Fe, Pt, Mg, O).
2.  The `DistillationConfig` is enabled (`enable: True`).
**Operating Procedures:**
1.  Execute the initialization script and launch Phase 1.
2.  Monitor the logs and output directories for generated structures.
**Expected Results (Acceptance Criteria):**
*   Subsystem structure pools (including random, strained, and defected structures) for pure elements and binary systems are automatically generated.
*   The DIRECT sampling method successfully reduces the number of structures to the specified sampling count (e.g., 1000).
*   MACE inference is performed, and only structures with an uncertainty below the `uncertainty_threshold` are extracted.
*   **Crucially, DFT (Quantum ESPRESSO) is never called.** A base potential (`base.yace`) is generated using the Lennard-Jones potential as a baseline.
*(Note: This scenario ensures that the foundation model effectively provides the "common sense" of the physical universe, drastically reducing the initial computational overhead.)*

### Scenario ID: UAT-PHASE2-001 (Priority: Medium)
**Title:** Phase 2 - Physical Validation and Automatic Retraining Verification
**Purpose:** To verify that if the constructed potential fails to meet the physical stability criteria, the system automatically increases the sampling density and triggers a self-healing (retraining) loop.
**Prerequisites:**
1.  A `base.yace` generated in Phase 1 exists.
2.  A potential intentionally constructed with extremely low sampling counts (to ensure low accuracy) is provided.
**Operating Procedures:**
1.  Launch the Validator and execute Phase 2.
**Expected Results (Acceptance Criteria):**
*   The system calculates the elastic constants, phonon dispersion, and Equations of State (EOS) for the stable phases.
*   When imaginary frequencies (instabilities) are detected in the phonon dispersion of the intentionally low-accuracy potential, the system **automatically expands the sampling density (or range) of Phase 1 and triggers a retraining process**.
*   A miniature MD stress test either completes successfully or, upon halting, outputs an Uncertainty Map (profiling the temperature dependence of the uncertainty).
*(Note: This scenario guarantees that the base potential is robust enough for the production environment before proceeding.)*

### Scenario ID: UAT-PHASE3-001 (Priority: High)
**Title:** Phase 3 - Thermal Noise Rejection and Intelligent Cluster Cutout
**Purpose:** To validate the core of the new architecture: the "Two-Tier Threshold" for noise tolerance and the clean, dangling-bond-free cluster cutout mechanism.
**Prerequisites:**
1.  Set up an MD simulation at a production scale (tens of thousands of atoms).
2.  Ensure `ActiveLearningThresholds` and `CutoutConfig` are properly configured.
**Operating Procedures:**
1.  Start the MD simulation.
2.  To simulate thermal noise, artificially spike the uncertainty of a single atom above the `threshold_call_dft` for only 1 to 2 steps (e.g., by manipulating data or setting a high temperature).
3.  Subsequently, introduce an unknown interface or defect structure into the system to cause a sustained increase in uncertainty.
**Expected Results (Acceptance Criteria):**
*   **Thermal Noise Tolerance:** The MD does **not** halt during the momentary spike in Step 2, proving the functionality of `smooth_steps`.
*   **Epicentre Identification:** When a sustained spike occurs in Step 3, the MD halts. Only the group of atoms exceeding `threshold_add_train` is identified as the "epicentre."
*   **Physical Repair Cutout:**
    *   The Core (`force_weight=1.0`) and Buffer (`force_weight=0.0`) regions are extracted correctly.
    *   **Crucially, while the Core atoms are fixed (Freezed), the Buffer region is pre-relaxed by MACE.**
    *   Auto-passivation applies dummy atoms (e.g., Hydrogen) to broken bonds on the surface, electrically neutralising the cluster.
*   **Reliable DFT Convergence:** The subsequent DFT calculation (SCF loop) on the extracted cluster completes normally without divergence, successfully acquiring the Ground Truth Force.

### Scenario ID: UAT-PHASE4-001 (Priority: High)
**Title:** Phase 4 - Hierarchical Finetuning and Seamless Resume
**Purpose:** To verify that incremental updates prevent catastrophic forgetting and that the MD can resume seamlessly without "rewinding" time.
**Prerequisites:**
1.  Scenario UAT-PHASE3-001 has been passed, and a small amount of clean DFT data has been acquired.
**Operating Procedures:**
1.  Monitor the flow from the learning process in Phase 4 to the resumption of the MD.
2.  Examine the energy changes in the MD log immediately after resuming.
**Expected Results (Acceptance Criteria):**
*   MACE is finetuned using the acquired DFT data.
*   The awakened MACE instantly generates thousands of surrogate data points.
*   **Prevention of Catastrophic Forgetting:** Delta Learning is executed using past data (replay buffer) and surrogate data, completing the learning in a short time ($O(1)$ computation cost) instead of performing batch retraining from scratch.
*   **Continuity Guarantee:** After the potential is updated, the MD does **not** start from "Step 0." Instead, it inherits the exact step number, coordinates, and velocities from immediately before the halt (proving Master-Slave Inversion).
*   **Soft Start:** For the first few steps after resuming, a Langevin heat bath (soft start) functions correctly, preventing non-continuous energy explosions (where the system "blows up").

### Scenario ID: UAT-HPC-001 (Priority: Medium)
**Title:** HPC Environment Robustness (Non-Functional) Stress Test
**Purpose:** To verify the system's resilience against forced job terminations (e.g., wall-time limits) and the automatic cleanup of artefacts.
**Prerequisites:**
1.  An actual HPC environment (e.g., Slurm) or an environment simulating parallel execution.
**Operating Procedures:**
1.  During an MD loop or surrogate generation task, intentionally terminate the main Python process forcefully (`kill -9`) to simulate a wall-time timeout.
2.  Resubmit the job (resume) in the same directory.
3.  Monitor the generation of massive `.wfc` (wavefunction) files in the background.
**Expected Results (Acceptance Criteria):**
*   **State Recovery:** Upon resubmission, the system does not start from the beginning. It recovers within seconds to minutes from the fine-grained SQLite/JSON checkpoint (e.g., mid-way through surrogate generation or immediately after DFT completion).
*   **Automatic Cleanup:** Massive dump files and `.wfc` files that are no longer needed after successful training/inference are automatically deleted or compressed by a daemon process, preventing storage exhaustion.

---

## 2. Behavior Definitions (Gherkin-style)

This section formalises the system's expected behaviour using BDD (Behaviour-Driven Development) conventions. This ensures absolute clarity for both developers and the automated testing framework regarding the exact states and transitions required by the architecture.

**Feature: Zero-Shot Distillation (Phase 1)**
As an active learning orchestrator,
I want to use a foundation model to generate an initial potential,
So that I can avoid expensive DFT calculations for baseline configurations.

*   **GIVEN** a quaternary system of Fe, Pt, Mg, and O is specified in the configuration,
*   **AND** `DistillationConfig.enable` is set to `True`,
*   **WHEN** the Phase 1 initialization process is triggered,
*   **THEN** the system should automatically generate combinatorial subsystem structures (pure and binary),
*   **AND** the `ActiveSetSelector` should filter these using DIRECT sampling to a maximum of 1000 structures,
*   **AND** the `MACEManager` should infer energies and filter out structures where uncertainty $> 0.05$,
*   **AND** the `PacemakerTrainer` should train a `base.yace` potential using Lennard-Jones Delta Learning without invoking the `QEDriver`.

**Feature: Intelligent Cutout and Physical Repair (Phase 3)**
As an MD engine encountering unknown configurations,
I want to extract only the uncertain local region and repair its boundaries,
So that the subsequent DFT calculation converges without dipole divergence.

*   **GIVEN** a massive MD simulation is running with a valid base potential,
*   **WHEN** the maximum uncertainty of the system exceeds `threshold_call_dft` ($0.05$) for $3$ consecutive steps,
*   **THEN** the MD should pause,
*   **AND** the system should identify atoms exceeding `threshold_add_train` ($0.02$) as the core epicentre,
*   **AND** the `extract_intelligent_cluster` utility should create a spherical cutout with `force_weight=1.0` for the core and `0.0` for the buffer,
*   **AND** the buffer region should be pre-relaxed by MACE while the core coordinates remain frozen,
*   **AND** fractional hydrogen atoms should be automatically added to any unbonded surface atoms to neutralise the cluster.

**Feature: Hierarchical Delta Learning and Seamless Resume (Phase 4)**
As a continuous learning system,
I want to incrementally update my potential and resume MD without rewinding,
So that I can observe long-timescale phenomena like phase transformations.

*   **GIVEN** a successfully converged DFT calculation for an extracted cluster is complete,
*   **WHEN** the Phase 4 learning sequence begins,
*   **THEN** the MACE model should be fine-tuned using the new ground truth force data,
*   **AND** the awakened MACE should generate a surrogate dataset of at least 1000 points around the halt state,
*   **AND** the `PacemakerTrainer` should execute Delta Learning using the surrogate data mixed with a 500-structure replay buffer from previous iterations,
*   **AND** upon completion, the MD should resume from the exact step, coordinates, and velocities it held at the moment of the halt, applying a Langevin heat bath for the first 100 steps.

**Feature: Robust Checkpointing and Recovery**
As a user running jobs on an HPC cluster,
I want the system to save fine-grained state data,
So that my progress is not lost when the scheduler kills my job due to time limits.

*   **GIVEN** a surrogate generation task is currently processing 1000 structures,
*   **WHEN** the Python process receives a `SIGKILL` (simulating a job termination) at structure 500,
*   **AND** the job is subsequently restarted in the same working directory,
*   **THEN** the system should read the SQLite/JSON checkpoint,
*   **AND** resume surrogate generation from structure 501, rather than starting from structure 1.

---

## 3. Tutorial Strategy

To ensure these UAT scenarios are not merely theoretical, they will be translated into executable tutorials using the `marimo` framework. This allows users to interactively verify the requirements, inspect the generated structures, and understand the workflow in a reproducible manner.

*   **Mock Mode vs. Real Mode:**
    *   For CI/CD and initial user onboarding (where Quantum ESPRESSO or massive GPUs might not be available), the tutorial will run in "Mock Mode." In this mode, the `QEDriver` will be patched to return dummy forces instantly, and the `MACEManager` will use a lightweight mock model.
    *   Users with full setups can toggle `REAL_MODE=True` in the tutorial to execute the actual heavy computations.
*   **Visualisation First:** The tutorial will heavily emphasise visualising the structural transformations. When Phase 3 executes, the `marimo` notebook will display the structure before cutout, the isolated cluster with dangling bonds, and the final pre-relaxed, passivated cluster ready for DFT.

## 4. Tutorial Plan

All test scenarios (Quick Start and Advanced HPC simulations) will be consolidated into a **SINGLE** executable file to minimise cognitive load for new users.

*   **Target File:** `tutorials/UAT_AND_TUTORIAL.py` (A Marimo Python/Text notebook).

This single file will guide the user sequentially through:
1.  **Initialization:** Setting up the configurations (`WorkflowConfig`, `CutoutConfig`, etc.).
2.  **Phase 1 Execution:** Triggering distillation and observing the output `base.yace`.
3.  **Phase 2 & 3 Simulation:** Starting a mock MD run, artificially injecting a high-uncertainty event, and visualizing the intelligent cutout process.
4.  **Phase 4 Resume:** Observing the Delta Learning process and verifying that the MD restarts at the correct step number.

## 5. Tutorial Validation

Before merging any architectural changes, the CI pipeline must validate that the `tutorials/UAT_AND_TUTORIAL.py` Marimo file executes correctly in Mock Mode without raising any unhandled exceptions. This ensures the documentation and the actual codebase remain perfectly synchronized.
