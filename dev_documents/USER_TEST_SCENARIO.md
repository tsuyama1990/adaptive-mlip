# PYACEMAKER v2.1.0 User Acceptance Testing (UAT) Scenarios
**Status**: DRAFT

## 1. Test Scenarios

### Scenario 1: Phase 1 - Zero-Shot Distillation & Baseline Construction
**Priority**: High
**Description**: Verify that a physically valid initial potential (with LJ Delta Learning applied) is constructed solely via MACE inference, without calling DFT.

*   **Prerequisites**:
    *   Input elements specified as a quaternary system (e.g., Fe, Pt, Mg, O).
    *   `DistillationConfig` is enabled (`enable: True`).
*   **Actions**:
    *   Execute the initialisation script to launch Phase 1.
    *   Monitor the logs and output directory.
*   **Expected Results (Acceptance Criteria)**:
    *   Combinatorial sub-system structure pools (random, strained, defective) are automatically generated.
    *   DIRECT sampling successfully reduces the number of structures to the specified sampling count (e.g., 1000).
    *   MACE inference retains only structures where uncertainty is below the `uncertainty_threshold`.
    *   `base.yace` is generated using a Lennard-Jones baseline **without a single call to DFT (QE)**.

### Scenario 2: Phase 2 - Physical Validation & Auto-Retraining
**Priority**: High
**Description**: Verify that the system automatically increases sampling density and retrains if the constructed potential fails physical stability criteria.

*   **Prerequisites**:
    *   `base.yace` from Phase 1 exists.
    *   A deliberately degraded potential is provided (simulating low sampling density in Phase 1).
*   **Actions**:
    *   Launch the Validator and execute Phase 2.
*   **Expected Results (Acceptance Criteria)**:
    *   Elastic constants, phonon dispersions, and EOS are calculated for stable phases.
    *   When imaginary frequencies (instabilities) are detected in the phonon dispersion of the degraded potential, **Phase 1 sampling density (or range) is automatically expanded, and retraining is triggered**.
    *   An Uncertainty Map (temperature dependence profile) is generated if the miniature MD stress test completes or halts.

### Scenario 3: Phase 3 - Thermal Noise Rejection & Intelligent Cutout
**Priority**: Critical
**Description**: Validate the "Two-Tier Threshold" for noise resistance and the clean, passivation-enabled cluster extraction algorithm.

*   **Prerequisites**:
    *   A production-scale MD setup (tens of thousands of atoms).
    *   `ActiveLearningThresholds` and `CutoutConfig` are properly configured.
*   **Actions**:
    *   Start the MD simulation.
    *   Artificially spike single-atom uncertainty above `threshold_call_dft` for 1-2 steps to simulate thermal noise.
    *   Introduce an unknown interface/defect structure to cause a sustained uncertainty rise.
*   **Expected Results (Acceptance Criteria)**:
    *   **Thermal Noise Resistance**: The MD does not halt during the 1-2 step spike (`smooth_steps` functionally proven).
    *   **Epicentre Identification**: The MD halts only during the sustained spike. Atoms exceeding `threshold_add_train` are identified as the "epicentre".
    *   **Physical Repair Cutout**:
        *   Core (`force_weight=1.0`) and Buffer (`force_weight=0.0`) are correctly extracted.
        *   **Core atoms are frozen while MACE pre-relaxes the Buffer region.**
        *   Broken bonds are auto-passivated (e.g., with H atoms), and the cluster is electronically neutralised.
    *   **Clean DFT Convergence**: The extracted cluster is processed by DFT (SCF loop) without divergence, yielding valid Ground Truth Forces.

### Scenario 4: Phase 4 - Hierarchical Fine-Tuning & Seamless Resume
**Priority**: Critical
**Description**: Verify incremental updates (preventing catastrophic forgetting) and Master-Slave continuous resume.

*   **Prerequisites**:
    *   Scenario 3 is complete, and sparse, clean DFT data has been acquired.
*   **Actions**:
    *   Monitor the Phase 4 training flow through to MD resumption.
    *   Check the MD energy logs immediately after resumption.
*   **Expected Results (Acceptance Criteria)**:
    *   The MACE model is fine-tuned using the acquired DFT data.
    *   The awakened MACE instantaneously generates thousands of surrogate data points.
    *   **Catastrophic Forgetting Prevention**: Delta Learning is executed incrementally using past data (Replay Buffer) and surrogate data, completing in $O(1)$ time rather than via full batch retraining.
    *   **Continuity Guarantee**: Following the potential update, MD **resumes from the exact halted step, coordinates, and velocities**, proving Master-Slave inversion.
    *   **Soft Start**: A short Langevin thermostat prevents unphysical energy explosions during the first few steps post-resume.

### Scenario 5: HPC Environment Robustness (Non-Functional)
**Priority**: Medium
**Description**: Verify tolerance against process kills and automatic artifact cleanup.

*   **Prerequisites**:
    *   An HPC environment (Slurm) or parallel execution emulation.
*   **Actions**:
    *   Intentionally kill the main Python process (`kill -9`) during MD or surrogate generation (simulating a Wall-time limit).
    *   Re-submit the job in the same directory.
    *   Monitor the generation of massive `.wfc` files.
*   **Expected Results (Acceptance Criteria)**:
    *   **State Recovery**: Upon re-submission, the system resumes within seconds/minutes from the last fine-grained SQLite/JSON checkpoint (e.g., mid-surrogate generation or post-DFT), not from scratch.
    *   **Auto-Cleanup**: Massive `.wfc` and dump files are automatically deleted or gzipped by a daemon process after successful training/inference.

---

## 2. Behaviour Definitions (Gherkin)

**Scenario 3.1: Thermal Noise Filtering**
**GIVEN** an active Molecular Dynamics simulation
**AND** the `ActiveLearningThresholds.smooth_steps` is set to 3
**WHEN** the maximum atomic uncertainty exceeds `threshold_call_dft` for only 1 or 2 consecutive steps
**THEN** the orchestrator shall ignore the spike as thermal noise
**AND** the MD simulation shall continue without halting.

**Scenario 3.2: Intelligent Cutout Passivation**
**GIVEN** a halted MD simulation
**AND** an epicentre atom has been identified
**WHEN** the cluster is extracted using `CutoutConfig`
**THEN** the system shall freeze atoms within `core_radius`
**AND** the system shall relax atoms within `buffer_radius` using MACE
**AND** the system shall identify broken bonds on the cluster surface and attach pseudo-atoms to neutralise the charge.

**Scenario 4.1: Master-Slave Resume**
**GIVEN** a freshly updated ACE potential from Phase 4
**AND** a saved LAMMPS restart file from the exact moment of the halt
**WHEN** the orchestrator signals LAMMPS to resume
**THEN** LAMMPS shall load the updated potential
**AND** LAMMPS shall read the restart file, restoring exact atomic coordinates and velocities
**AND** LAMMPS shall apply a `fix langevin` soft start for the first $N$ steps before returning to the NVE/NVT ensemble.

---

## 3. Tutorial Strategy

To ensure reproducible verification of these requirements, the UAT scenarios will be executable as a tutorial.

### Mock Mode vs. Real Mode Strategy
*   **Real Mode**: Requires active internet connections (for downloading MACE weights), Quantum Espresso binaries (`pw.x`), LAMMPS binaries, and Pacemaker installations. This takes significant time (hours) to execute fully.
*   **Mock Mode (CI Execution)**: For rapid UAT verification without heavy computational infrastructure, a Mock Mode will be implemented. When enabled, `QEDriver` returns dummy forces and energies, `MACEManager` bypasses PyTorch inference to return mathematically scaled vectors, and `LammpsEngine` uses simplified mock configuration scripts instead of actual C++ binaries.

### Tutorial Plan
A **SINGLE** Marimo Text/Python notebook file will be created at `tutorials/UAT_AND_TUTORIAL.py`.
This file will contain all scenarios (Quick Start and Advanced features) within one executable environment, allowing users to verify UAT Scenarios 1 through 4 visually and programmatically using `marimo edit tutorials/UAT_AND_TUTORIAL.py`.

### Tutorial Validation
The Marimo file will be validated in Mock Mode to ensure that the sequential cell execution passes without raising exceptions, thereby verifying that the API contracts, domain model instantiations, and state transitions function as defined in the architecture.