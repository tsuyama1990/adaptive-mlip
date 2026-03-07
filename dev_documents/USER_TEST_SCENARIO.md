# PYACEMAKER v2.1.0 User Acceptance Testing (UAT) and Tutorial Scenarios

This document outlines the User Acceptance Testing (UAT) scenarios designed from a researcher's (user's) perspective to validate the quality and usability of the PyAceMaker NextGen Architecture (Version 2.1.0). These scenarios also serve as a tutorial strategy to guide new users through the advanced capabilities of the system.

## 1. Tutorial Strategy

To ensure reproducibility and ease of use, all scenarios defined below will be compiled into a single interactive tutorial file.

*   **Executable Tutorial:** We will create a single Marimo notebook file named `tutorials/UAT_AND_TUTORIAL.py`. This single file will contain both "Quick Start" and "Advanced" scenarios, allowing researchers to easily execute and verify the workflow interactively using `marimo edit tutorials/UAT_AND_TUTORIAL.py`.
*   **Mock Mode vs. Real Mode:** To facilitate rapid testing in CI environments or on laptops without access to large HPC clusters or Quantum Espresso installations, the tutorial will support a "Mock Mode". In Mock Mode, the `DFTManager` is replaced with a dummy oracle that returns fixed forces, allowing the entire pipeline to execute in minutes rather than days. Real Mode can be toggled via a configuration flag.
*   **Visual Validation:** The tutorial will output `.xyz` files at critical steps (e.g., before cutout, after cutout, after MACE relaxation, after passivation) so users can visually inspect the physical validity of the operations using tools like OVITO.

## 2. Test Scenarios

### Scenario ID: UAT-01
**Title:** Phase 1 - Zero-Shot Distillation and Baseline Construction
**Priority:** High
**Objective:** Verify that a physically reasonable initial potential (using LJ Delta Learning) is constructed solely through MACE inference, without calling DFT.

*   **Pre-conditions:**
    *   The input elements are specified as a 4-element system (e.g., Fe, Pt, Mg, O).
    *   The `DistillationConfig` is enabled (`enable: True`).
*   **Action:**
    1.  Execute the initialization script to launch Phase 1.
    2.  Monitor the logs and the output directory.
*   **Expected Results (Acceptance Criteria):**
    *   The system automatically generates structural pools for all single-element and binary sub-systems, including random, strained, and defected structures.
    *   DIRECT sampling successfully reduces the pool to the requested sampling size (e.g., 1000 structures) while maximizing diversity.
    *   MACE inference is performed, and only structures with uncertainty below the `uncertainty_threshold` are extracted.
    *   **Crucially, DFT (Quantum Espresso) is never called.**
    *   The `PacemakerTrainer` generates `base.yace` using the extracted data, successfully applying LJ Delta Learning as the baseline.

### Scenario ID: UAT-02
**Title:** Phase 2 - Physical Validation and Auto-Retraining
**Priority:** High
**Objective:** Confirm that if the constructed potential fails minimum physical stability criteria, the system automatically expands sampling and triggers retraining.

*   **Pre-conditions:**
    *   `base.yace` from Phase 1 exists.
    *   For testing purposes, the potential is intentionally degraded by severely restricting the sampling density in Phase 1.
*   **Action:**
    1.  Launch the `Validator` to execute Phase 2.
*   **Expected Results (Acceptance Criteria):**
    *   The Validator attempts to calculate elastic constants, phonon dispersions, and the Equation of State (EOS) for stable phases.
    *   Upon detecting an instability (e.g., imaginary frequencies in the phonon dispersion) in the intentionally degraded potential, **the system automatically triggers an expansion of the Phase 1 sampling density and re-initiates the training loop.**
    *   A miniature MD stress test completes or halts, successfully outputting an Uncertainty Map (a profile of uncertainty versus temperature).

### Scenario ID: UAT-03
**Title:** Phase 3 - Thermal Noise Exclusion and Intelligent Cluster Extraction
**Priority:** Critical
**Objective:** Validate the core paradigm shift: the two-tier threshold for noise resilience, and the generation of a clean, passivated cluster free of dangling bonds.

*   **Pre-conditions:**
    *   A production-scale MD simulation (tens of thousands of atoms) is set up.
    *   `ActiveLearningThresholds` and `CutoutConfig` are configured.
*   **Action:**
    1.  Start the MD simulation.
    2.  Simulate thermal noise by artificially spiking the uncertainty of a single atom above `threshold_call_dft` for only 1 or 2 steps.
    3.  Simulate a true unknown physical event by introducing an unknown defect or interface, causing a sustained uncertainty spike over many steps.
*   **Expected Results (Acceptance Criteria):**
    *   **Thermal Noise Resilience:** The MD simulation does not halt during the artificial 1-2 step spike (proving the `smooth_steps` logic works).
    *   **Epicenter Identification:** When the sustained spike occurs, the system halts. Only the specific atoms exceeding `threshold_add_train` are identified as the "epicenter".
    *   **Physical Repair Cutout:**
        *   The core region (`force_weight=1.0`) and buffer region (`force_weight=0.0`) are correctly extracted spherically.
        *   **With the core atoms strictly frozen, MACE successfully pre-relaxes the buffer region.**
        *   Auto-passivation correctly identifies broken bonds at the surface and adds dummy atoms (e.g., fractional H) to electrically neutralize the cluster.
    *   **DFT Convergence:** The resulting extracted cluster allows the DFT calculation (SCF loop) to converge without divergence, successfully acquiring Ground Truth Forces.

### Scenario ID: UAT-04
**Title:** Phase 4 - Hierarchical Fine-Tuning and Seamless Resume
**Priority:** Critical
**Objective:** Verify that incremental update prevents catastrophic forgetting and that the MD simulation resumes smoothly without rewinding time.

*   **Pre-conditions:**
    *   Scenario UAT-03 has completed, yielding a small amount of clean DFT data.
*   **Action:**
    1.  Monitor the Phase 4 training process and the subsequent MD resumption.
    2.  Check the energy logs of the MD immediately after resumption.
*   **Expected Results (Acceptance Criteria):**
    *   MACE is briefly fine-tuned using the newly acquired DFT data.
    *   The awakened MACE instantly generates thousands of surrogate data points around the halt state.
    *   **Prevention of Catastrophic Forgetting:** The system executes **Delta Learning** using the surrogate data, the true DFT anchor, and a replay buffer of historical data, completing the training in O(1) time without rebuilding the entire dataset from scratch.
    *   **Continuity Guarantee:** Following the potential update, the MD simulation **resumes from the exact step number, coordinates, and velocities where it halted** (proving the Master-Slave inversion).
    *   **Soft Start:** A Langevin thermostat (or similar soft-start mechanism) functions during the first few resumed steps, preventing unphysical energy explosions due to the potential switch.

### Scenario ID: UAT-05
**Title:** Non-Functional - HPC Robustness and Cleanup
**Priority:** Medium
**Objective:** Verify resilience against forced job terminations and the automatic cleanup of massive artifacts.

*   **Pre-conditions:**
    *   Running in an HPC-like environment (or using process-level emulation).
*   **Action:**
    1.  During a surrogate generation task or MD loop, intentionally kill the main Python process using `kill -9` (simulating a wall-time timeout).
    2.  Resubmit the job in the same directory.
    3.  Monitor the generation of large `.wfc` (wavefunction) files.
*   **Expected Results (Acceptance Criteria):**
    *   **State Recovery:** Upon resubmission, the system does not start from the beginning. Using the fine-grained SQLite/JSON checkpoints, it recovers its state within seconds and resumes from the exact micro-task it was executing (e.g., midway through surrogate generation or immediately after a DFT completion).
    *   **Auto-Cleanup:** Large artifact files (like `.wfc` or massive dump files) that are no longer needed after successful training or inference are automatically deleted or compressed by a background daemon, ensuring storage limits are not exceeded.

## 3. Behavior Definitions (Gherkin)

**Feature:** Zero-Shot Baseline Construction
> **GIVEN** a 4-element system definition and DistillationConfig is enabled
> **WHEN** Phase 1 initialization is triggered
> **THEN** structural pools are generated
> **AND** structures are filtered using MACE uncertainty
> **AND** a baseline potential is trained using LJ Delta Learning without any DFT calls.

**Feature:** Two-Tier Noise Filtering
> **GIVEN** an active MD simulation with `threshold_call_dft` set to 0.05 and `smooth_steps` set to 3
> **WHEN** a single atom's uncertainty spikes to 0.08 for exactly 1 step
> **THEN** the simulation does not halt
> **WHEN** the uncertainty remains at 0.08 for 4 consecutive steps
> **THEN** the simulation halts
> **AND** only atoms with uncertainty above `threshold_add_train` are selected for extraction.

**Feature:** Intelligent Extraction with Pre-relaxation
> **GIVEN** an identified epicenter in a halted MD simulation
> **WHEN** the cluster is extracted
> **THEN** a core region and a buffer region are defined
> **AND** the core atoms are constrained (frozen)
> **AND** MACE relaxes the coordinates of the buffer atoms
> **AND** dummy atoms are added to passivate surface dangling bonds.

**Feature:** Seamless MD Resume
> **GIVEN** a halted MD simulation at step 500,000 due to high uncertainty
> **WHEN** the new ACE potential is incrementally trained and loaded
> **THEN** the MD simulation resumes exactly at step 500,001
> **AND** the atomic coordinates, velocities, and thermostat state are identical to the moment of the halt.

## 4. Tutorial Plan

As stated in the Tutorial Strategy, we will create a **SINGLE** executable file to validate these scenarios.

**File:** `tutorials/UAT_AND_TUTORIAL.py`

This Marimo file will be structured as follows:
1.  **Introduction & Setup:** Explains the NextGen architecture and initializes the configuration objects (Pydantic models).
2.  **Scenario 1: Phase 1 Distillation:** Executes the zero-shot baseline construction interactively.
3.  **Scenario 2: Validation Stress Test:** Runs the physical validator on the generated baseline.
4.  **Scenario 3: The Halt Event:** Simulates an MD run, injects a fake high-uncertainty event to trigger a halt, and demonstrates the intelligent cluster extraction (outputting images of the passivated cluster).
5.  **Scenario 4: Incremental Update & Resume:** Runs the Pacemaker training using a replay buffer and shows how the MD state is preserved for resumption.

## 5. Tutorial Validation
Before submitting the code, developers must run:
```bash
uv run marimo run tutorials/UAT_AND_TUTORIAL.py
```
This command must execute successfully from top to bottom, proving that the API contracts defined in the architecture are sound and that the user experience is flawless.