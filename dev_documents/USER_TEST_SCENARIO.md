# PYACEMAKER v2.1.0 User Acceptance Testing (UAT) and Tutorial Scenarios

This document outlines the exhaustive User Acceptance Testing (UAT) scenarios designed from a researcher's (user's) perspective to rigorously validate the quality, usability, and physical correctness of the PyAceMaker NextGen Architecture (Version 2.1.0). These scenarios also serve as a comprehensive tutorial strategy to guide new users through the advanced capabilities of the system.

## 1. Tutorial Strategy

To ensure absolute reproducibility and ease of use, all scenarios defined below will be compiled into a single interactive, executable tutorial file.

*   **Executable Tutorial:** We will create a single Marimo notebook file named `tutorials/UAT_AND_TUTORIAL.py`. This single file will contain both "Quick Start" and "Advanced" scenarios, allowing researchers to easily execute, modify, and verify the workflow interactively using `marimo edit tutorials/UAT_AND_TUTORIAL.py`.
*   **Mock Mode vs. Real Mode:** To facilitate rapid testing in CI environments, or on personal laptops without access to large HPC clusters, GPUs, or Quantum Espresso installations, the tutorial will inherently support a "Mock Mode". In Mock Mode, the `DFTManager` is replaced with a dummy oracle that returns fixed analytical forces (e.g., using a simple Lennard-Jones definition), allowing the entire sophisticated pipeline to execute in minutes rather than days. Real Mode can be toggled via a simple configuration flag.
*   **Visual Validation:** The tutorial will output standard `.xyz` trajectory files at critical algorithmic steps (e.g., before cutout, after cutout, after MACE relaxation, after fractional passivation). This allows users to visually inspect the physical validity of the operations and geometric boundary constraints using external tools like OVITO or VMD.

## 2. Test Scenarios

### Scenario ID: UAT-01
**Title:** Phase 1 - Zero-Shot Distillation and Baseline Construction
**Priority:** High
**Objective:** Verify that a physically reasonable initial potential (using LJ Delta Learning) is constructed solely through MACE inference, without calling DFT, and that the resulting `base.yace` is stable at short atomic distances.

*   **Pre-conditions:**
    *   The input elements are specified as a complex 4-element system (e.g., Fe, Pt, Mg, O).
    *   The `DistillationConfig` is enabled (`enable: True`).
    *   The `MACEManager` is configured to run in memory-resident mode.
*   **Action:**
    1.  Execute the initialization script to launch Phase 1.
    2.  Monitor the logs, structure pools, and the output directory for `base.yace`.
*   **Expected Results (Acceptance Criteria):**
    *   The system automatically generates structural pools for all single-element and binary sub-systems, including random, strained, and defected structures (vacancies).
    *   DIRECT sampling (KDTree-assisted D-Optimality) successfully reduces the massive pool to the requested sampling size (e.g., 1000 structures) while maximizing diversity in feature space.
    *   MACE inference is performed in memory, and only structures with uncertainty strictly below the `uncertainty_threshold` are extracted into the training dataset.
    *   **Crucially, DFT (Quantum Espresso) is never called or initialized.**
    *   The `PacemakerTrainer` generates `base.yace` using the extracted data, successfully applying LJ Delta Learning as the mathematical baseline to prevent atomic overlap crashes.

### Scenario ID: UAT-02
**Title:** Phase 2 - Physical Validation and Auto-Retraining
**Priority:** High
**Objective:** Confirm that if the constructed potential fails minimum physical stability criteria (e.g., imaginary phonons), the system automatically detects this, expands sampling, and triggers a state-machine retraining loop.

*   **Pre-conditions:**
    *   `base.yace` from Phase 1 exists.
    *   For testing purposes, the potential is intentionally degraded by severely restricting the sampling density or altering the LJ parameters in Phase 1 to induce instability.
*   **Action:**
    1.  Launch the `Validator` module to execute Phase 2 via the orchestrator.
*   **Expected Results (Acceptance Criteria):**
    *   The Validator attempts to calculate elastic constants, phonon dispersions, and the Equation of State (EOS) for all stable crystal phases.
    *   Upon detecting a severe instability (e.g., imaginary frequencies in the acoustic branches of the phonon dispersion) in the intentionally degraded potential, **the system automatically triggers a rollback to the `DISTILLING` state, expanding the Phase 1 sampling density and re-initiating the entire training loop.**
    *   A miniature MD stress test (e.g., NPT ensemble) completes or halts, successfully outputting an Uncertainty Map (a profile of MACE uncertainty variance versus simulation temperature).

### Scenario ID: UAT-03
**Title:** Phase 3 - Thermal Noise Exclusion and Intelligent Cluster Extraction
**Priority:** Critical
**Objective:** Validate the core paradigm shift: the two-tier threshold for noise resilience (`smooth_steps`), and the generation of a clean, passivated cluster free of dangling bonds utilizing KDTree extraction.

*   **Pre-conditions:**
    *   A production-scale MD simulation (tens of thousands of atoms) is set up and thermalized.
    *   `ActiveLearningThresholds` and `CutoutConfig` are correctly configured.
*   **Action:**
    1.  Start the MD simulation.
    2.  Simulate severe thermal noise by artificially spiking the uncertainty array of a single atom above `threshold_call_dft` for only 1 or 2 integration steps.
    3.  Simulate a true unknown physical event by introducing an unknown defect or interface collision, causing a sustained uncertainty spike over $N >$ `smooth_steps`.
*   **Expected Results (Acceptance Criteria):**
    *   **Thermal Noise Resilience:** The MD simulation does not halt during the artificial 1-2 step spike (proving the `smooth_steps` logic correctly filters outliers).
    *   **Epicenter Identification:** When the sustained spike occurs, the system formally halts. Only the specific atoms exceeding the lower `threshold_add_train` are identified as the extraction "epicenter".
    *   **Physical Repair Cutout (KDTree):**
        *   The core region (`force_weight=1.0`) and buffer region (`force_weight=0.0`) are correctly extracted spherically using $O(\log N)$ KDTree queries.
        *   **With the core atoms strictly frozen (`FixAtoms`), MACE successfully pre-relaxes the buffer region.**
        *   Auto-passivation correctly identifies broken coordination bonds at the surface and injects dummy atoms (e.g., fractional H) to electrically neutralize the cluster. The minimum distance between dummy atoms and the core is rigorously $> 0.8 \AA$.
    *   **DFT Convergence:** The resulting extracted, passivated cluster allows the DFT calculation (SCF loop) to converge efficiently without electron density divergence, successfully acquiring Ground Truth Forces for the core.

### Scenario ID: UAT-04
**Title:** Phase 4 - Hierarchical Fine-Tuning and Seamless Resume
**Priority:** Critical
**Objective:** Verify that incremental update prevents catastrophic forgetting (via replay buffers) and that the MD simulation resumes smoothly without rewinding simulation time or losing thermostat state.

*   **Pre-conditions:**
    *   Scenario UAT-03 has completed, yielding a small amount of highly valuable, clean DFT data.
    *   A historical `training_history.extxyz` exists.
*   **Action:**
    1.  Monitor the Phase 4 training process and the subsequent MD resumption via LAMMPS.
    2.  Check the thermodynamic energy logs of the MD immediately after the resumption command.
*   **Expected Results (Acceptance Criteria):**
    *   MACE is briefly fine-tuned using the newly acquired DFT data via the `FinetuneManager`.
    *   The awakened MACE instantly generates thousands of surrogate data points in a localized spatial grid around the halt state.
    *   **Prevention of Catastrophic Forgetting:** The system executes **Delta Learning** using the surrogate data, the true DFT anchor, and exactly `replay_buffer_size` structures drawn from the historical data, completing the training in O(1) time without rebuilding the entire massive dataset from scratch.
    *   **Continuity Guarantee:** Following the potential update, the MD simulation **resumes from the exact step number, coordinates, and atomic velocities where it halted** (proving the Master-Slave inversion or perfect `.restart` execution).
    *   **Soft Start:** A Langevin thermostat (or `velocity scale`) mechanism functions during the first few resumed integration steps, preventing unphysical thermodynamic energy explosions due to the updated potential surface.

### Scenario ID: UAT-05
**Title:** Non-Functional - HPC Robustness, Repository Pattern, and Cleanup
**Priority:** Medium
**Objective:** Verify resilience against forced job terminations (using the JsonLinesRepository) and the automatic, asynchronous cleanup of massive computational artifacts.

*   **Pre-conditions:**
    *   Running in an HPC-like environment (or using process-level emulation via `multiprocessing`).
*   **Action:**
    1.  During a massive surrogate generation task or MD loop, intentionally kill the main Python orchestrator process using `kill -9` (simulating a strict SLURM wall-time timeout).
    2.  Resubmit the identical job in the exact same directory.
    3.  Monitor the generation of massive `.wfc` (wavefunction) files during a DFT fallback.
*   **Expected Results (Acceptance Criteria):**
    *   **State Recovery (Repository Pattern):** Upon resubmission, the system does not foolishly start from the beginning. Using the fine-grained `.jsonl` state checkpoints, it recovers its exact FSM state within seconds and resumes from the exact micro-task it was executing (e.g., midway through surrogate generation or immediately after a DFT completion).
    *   **Auto-Cleanup Daemon:** Massive artifact files (like `.wfc` or huge `.dump` trajectories) that are no longer needed after successful training or inference are automatically deleted or compressed (`tar -czf`) by an isolated background Python daemon, guaranteeing storage quotas are not exceeded without blocking the main event loop.

## 3. Behavior Definitions (Gherkin)

The following Gherkin scenarios define the explicit contracts that the system must uphold during automated integration testing.

**Feature: Zero-Shot Baseline Construction**
> **GIVEN** a 4-element system definition and DistillationConfig is enabled with memory-resident MACE
> **WHEN** Phase 1 initialization is triggered by the Orchestrator
> **THEN** structural pools are generated via combinatorial and defect algorithms
> **AND** structures are strictly filtered using the MACE uncertainty array against `uncertainty_threshold`
> **AND** a baseline potential is trained using LJ Delta Learning without any DFT engine calls being invoked.

**Feature: Two-Tier Noise Filtering and Smoothing**
> **GIVEN** an active MD simulation with `threshold_call_dft` set to 0.05 and `smooth_steps` set to 3
> **WHEN** a single atom's evaluated uncertainty spikes to 0.08 for exactly 1 integration step
> **THEN** the simulation does not halt and continues integration
> **WHEN** the evaluated uncertainty remains at 0.08 for 4 consecutive integration steps
> **THEN** the simulation explicitly halts and writes a restart state
> **AND** only atoms with uncertainty strictly above the secondary `threshold_add_train` are selected for extraction.

**Feature: Intelligent Extraction with KDTree and Pre-relaxation**
> **GIVEN** an identified epicenter in a halted MD simulation
> **WHEN** the intelligent cluster is extracted via KDTree querying
> **THEN** a core region and a buffer region are defined spherically
> **AND** the core atoms are constrained (frozen) using ASE `FixAtoms`
> **AND** the memory-resident MACE relaxes the coordinates of the buffer atoms via LBFGS
> **AND** dummy atoms are added to passivate surface dangling bonds, maintaining a minimum $0.8 \AA$ clearance from the core.

**Feature: Seamless MD Resume with Delta Learning**
> **GIVEN** a halted MD simulation at step 500,000 due to sustained high uncertainty
> **WHEN** the new ACE potential is incrementally trained using a replay buffer and loaded
> **THEN** the MD simulation resumes exactly at step 500,001
> **AND** the atomic coordinates, velocities, and thermostat state are mathematically identical to the moment of the halt.

**Feature: FSM Checkpoint Recovery**
> **GIVEN** the Orchestrator is in the `EXTRACTING` state
> **WHEN** the process receives a SIGKILL signal
> **THEN** the state is preserved in the `.jsonl` repository
> **WHEN** the process is restarted
> **THEN** it resumes operation immediately in the `EXTRACTING` state without rewinding to `DISTILLING`.

## 4. Tutorial Plan

As stated in the Tutorial Strategy, we will create a **SINGLE** executable interactive file to validate these complex scenarios.

**File:** `tutorials/UAT_AND_TUTORIAL.py`

This Marimo notebook file will be structured sequentially as follows:
1.  **Introduction & Setup:** Explains the NextGen architecture and initializes the strongly-typed configuration objects (Pydantic models like `ActiveLearningThresholds`).
2.  **Scenario 1: Phase 1 Distillation:** Executes the zero-shot baseline construction interactively, rendering the generated `extxyz` pool using `marimo.ui.slider` for visualization.
3.  **Scenario 2: Validation Stress Test:** Runs the physical validator on the generated baseline, plotting the resulting phonon dispersion bands.
4.  **Scenario 3: The Halt Event:** Simulates a small MD run, injects a fake high-uncertainty event array to trigger a halt, and demonstrates the intelligent cluster extraction (outputting 3D interactive renders of the passivated cluster).
5.  **Scenario 4: Incremental Update & Resume:** Runs the Pacemaker training using a tiny mock replay buffer and shows how the MD state dictionary is perfectly preserved for resumption.
6.  **Scenario 5: State Recovery:** Demonstrates reading a mock `.jsonl` log to recover a failed state.

## 5. Tutorial Validation
Before developers submit code related to this architecture, they must run:
```bash
uv run marimo run tutorials/UAT_AND_TUTORIAL.py
```
This command must execute successfully from top to bottom, proving that the strict API contracts defined in the architecture are sound, the physics bounds are respected, and that the user experience is flawless.