# PYACEMAKER NextGen Architecture Specification (PRD)
**Version**: 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date**: 2026-02-28
**Status**: DRAFT

## 1. Project Background and Purpose

### 1.1 Limitations of the Current System (Phase 01)
The basic orchestration implemented in Phase 01 established the foundation for the Active Learning loop. However, when deploying to practical HPC scales (Molecular Dynamics simulations involving tens of thousands to millions of atoms over long durations), several critical physical and systemic constraints became apparent:

1.  **MD Time-Continuity Break**: When the simulation halts because the uncertainty ($\gamma$) exceeds the threshold, the MD restarts from the initial structure after retraining. This design prevents the system from reaching long-timescale physical phenomena such as phase transformations or diffusion.
2.  **Thermal Noise False Positives**: Because a single uncertainty threshold determines when to halt, the system overreacts to instantaneous spikes caused by harmless thermal vibrations during MD. This triggers unnecessary calculations and infinite loops.
3.  **Physical Divergence in Local Extraction**: Simply extracting a highly uncertain region (cluster) from a massive system into a vacuum for DFT creates numerous dangling bonds. This leads to charge imbalances and dipole moment divergence, causing the DFT SCF loop to fail or the model to learn unphysical "garbage" electronic states.
4.  **Sluggish Batch Retraining & Catastrophic Forgetting**: Retraining the entire dataset from scratch every step causes an explosion in computational cost ($O(N)$). Furthermore, continuously adding only erroneous structures degrades the predictive accuracy of stable bulk structures.
5.  **System Fragility with LAMMPS**: C++ layer crashes in LAMMPS (e.g., "Lost atoms") take down the Python main process. The entire orchestrator stops without saving the system state.

### 1.2 Extracting Best Practices from Prior Research (FLARE)
Based on an analysis of the FLARE architecture (developed by Harvard University and others), four paradigm shifts must be introduced into this system:

1.  **Master-Slave Inversion (Inversion of Control)**: Python should no longer drive LAMMPS linearly. Instead, Python should be subordinated within the LAMMPS C++ loop (via `fix python/invoke` or periodic callbacks), or robust restart mechanisms should be used. This allows the system to seamlessly pause, update the potential, and resume without rewinding MD time.
2.  **Two-Tier Thresholds**: Separate the threshold for calling DFT (`threshold_call_dft`) from the threshold for adding data to the training set (`threshold_add_train`). This builds tolerance against thermal noise.
3.  **Global Calculation, Local Learning**: Compute the electronic state in a physically stable environment, but only use the "Force" on the highly uncertain "core atoms" to update the learning model.
4.  **Incremental Update**: Replace batch retraining with Delta Learning (incremental updating using past weights as initial values) to keep computational costs at $O(1)$.

### 1.3 Core Philosophy of the NextGen Architecture
By integrating the challenges above with the lessons from FLARE, we define a NextGen workflow centred on **Hierarchical Distillation**, **Intelligent Cluster Extraction**, and **Seamless Master-Slave MD Resume**.
To enable MD simulations of millions of atoms, we adopt the approach of "physically repairing and extracting" rather than "not extracting at all." By maximising the distillation and fine-tuning of foundation models (like MACE) across the system, we aim to minimise the high-cost DFT calculations while achieving near-DFT accuracy in targeted regions.

---

## 2. 4-Phase Hierarchical Distillation Workflow (Core Workflow)

The system executes the following four sequential phases for a target system (e.g., a quaternary Fe-Pt-Mg-O system):

### Phase 1: Zero-Shot Distillation & Baseline Construction
**Purpose**: Extract the generalisation capabilities ("universal common sense") of foundation models like MACE-MP-0 without calling DFT, and bake them into the ACE potential.

1.  **Combinatorial Exploration**: Automatically define unary and binary sub-systems from the input elements. Generate a massive structural pool including random structures, strains, rattles, high-temperature snapshots, stoichiometry variations, and defects (vacancies, interstitials, anti-sites).
2.  **Information Maximisation via DIRECT Sampling**: Utilise the existing `ActiveSetSelector` (DIRECT sampling / D-Optimality) to extract the most informative and diverse structures (hundreds to thousands) from the pool. This eliminates redundancy and minimises downstream costs.
3.  **Confidence Filtering**: Pass the extracted structures to `MACEManager` (the foundation model Oracle). Only retain structures where the MACE uncertainty falls below a specified threshold (safe structures where MACE is confident) as ground truth data.
4.  **Baseline ACE Training (LJ Delta Learning)**: Use the high-quality, confidence-filtered data across a broad chemical space to train a baseline many-body potential (`base.yace`) via `PacemakerTrainer`. Apply Delta Learning from a Lennard-Jones (LJ) potential as a default configuration to prevent unphysical atomic overlap at short distances.

### Phase 2: Validation & Stress Test
**Purpose**: Verify that the foundational potential built in Phase 1 guarantees minimum physical stability in the production environment.

1.  **Physical Property Inspection**: Launch the Validator to compute elastic constants (Born stability criteria), phonon dispersions (absence of imaginary frequencies), and equations of state (EOS) for the stable phases of each sub-system. If it fails, automatically increase the Phase 1 sampling density and retrain.
2.  **Miniature MD Stress Test**: Create a scaled-down production environment (e.g., a slab model of a few thousand atoms) and run MD with the new potential. Profile where halts occur or at what temperatures uncertainty rises (Uncertainty Map).

### Phase 3: Intelligent Cutout & Passivation
**Purpose**: When a massive MD encounters unknown local structures (interfaces, defects, collisions) and halts, automatically generate physically valid, clean clusters that DFT can process.

1.  **Epicentre Identification (Two-Tier Evaluation)**: Only trigger a Halt if the system's maximum uncertainty exceeds `threshold_call_dft` for several consecutive steps (filtering out thermal noise). Then, evaluate site-specific uncertainty and identify atoms exceeding `threshold_add_train` as the "epicentre".
2.  **Spherical Cutout & Weighting**: Extract a spherical region from the epicentre. Assign `force_weight = 1.0` to atoms within radius $R_{core}$, and `force_weight = 0.0` to atoms within radius $R_{buffer}$. Place the extracted cluster securely in a Periodic Boundary Condition (PBC) cell with a vacuum layer.
3.  **Boundary Pre-relaxation via MACE**: Freeze the coordinates of the core atoms. Use MACE to relax only the buffer atoms to eliminate unnatural bonding strains caused by the extraction.
4.  **Auto-Passivation**: Detect broken bonds at the outer edge of the buffer (especially for oxides like Mg/O) and automatically add dummy atoms (e.g., Fractional Hydrogen) to neutralise the cluster's charge and dipole moment.
5.  **Clean DFT Calculation**: Pass the physically and electrically stabilised cluster to the `QEDriver`. Use self-healing features (smearing extension, mixing beta adjustment) to ensure SCF convergence and obtain the Ground Truth Force for the core atoms.

### Phase 4: Hierarchical Fine-Tuning (Delta Learning)
**Purpose**: Chain-update MACE and ACE using the sparse, valuable DFT data, then safely resume MD.

1.  **Awaken MACE (Finetune MACE)**: Fine-tune the MACE foundation model using the acquired DFT data, allowing it to fully understand the specific interfacial physics of the target system.
2.  **Explosive Surrogate Data Generation**: Use the awakened MACE as an Oracle to instantaneously generate and infer thousands of surrogate data points in the phase space surrounding the halt event (via random displacements or micro-MD).
3.  **Incremental ACE Update**: Input the massive surrogate dataset and the anchor DFT data into `PacemakerTrainer`. To prevent computational explosion, perform incremental Delta Learning from the previous potential state, mixing in a Replay Buffer.
4.  **Seamless Resume (Master-Slave Resume)**: Load the updated potential and safely resume MD from the exact step, time, coordinates, and velocities where it halted.

---

## 3. Module Specifications

### 3.1. `pyacemaker.utils.extraction` (Major Extension)
The core module for cluster extraction and passivation.

*   **`extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms`**
    *   **Input**: A massive ASE `Atoms` object and a list of atom indices exceeding `threshold_add_train`.
    *   **Process**:
        *   Spherical extraction using neighbor lists ($R_{core}$ and $R_{buffer}$).
        *   Assign `force_weight` arrays.
        *   `_pre_relax_buffer`: Fix core with `ase.constraints.FixAtoms` and relax buffer via MACE LBFGS.
        *   `_passivate_surface`: Detect dangling bonds based on electronegativity and radii, adding H or pseudo-atoms (`force_weight=0.0`).
    *   **Output**: A computable `Atoms` object with PBC, vacuum, and passivation.

### 3.2. `pyacemaker.core.oracle` (Multi-Tiering)
Abstracts the Oracle to transparently handle MACE and DFT.

*   **`MACEManager(BaseOracle)`**: Wrapper to execute MACE-MP-0 inference (GPU supported). Must output energy, forces, and uncertainties based on ensemble variance or latent distance.
*   **`TieredOracle(BaseOracle)`**: Manages query routing. Evaluates structures with `MACEManager` first. Only falls back to `QEDriver` (DFT) if uncertainty exceeds the specified threshold.

### 3.3. `pyacemaker.core.engine` (LAMMPS & Seamless Resume)
A robust engine that survives LAMMPS crashes and ensures time continuity.

*   **Process Isolation & `read_restart`**: If LAMMPS crashes (e.g., lost atoms), the Python main loop survives. MD resumes seamlessly by inheriting velocity distributions and ensemble states perfectly from periodically saved `.restart` files.
*   **Soft Start Protocol**: To prevent catastrophic system explosions from energy discontinuities upon potential change, the Python generator automatically injects a short, heavily damped Langevin thermostat (`fix langevin`) for the first $N$ steps after resuming to thermalise the system before restoring the original ensemble.

### 3.4. `pyacemaker.core.trainer` (Pacemaker & MACE Finetune)
*   **`FinetuneManager`**: Wraps the short-duration training of the MACE PyTorch readout layer using clean DFT datasets.
*   **Incremental Update for `PacemakerTrainer`**: Mitigates batch training cost explosion by inheriting previous potential states and mixing a fixed-size Replay Buffer (randomly sampled from `training_history.extxyz`) with the current dataset. Automatically generates `input.yaml` settings for LJ Delta Learning.

---

## 4. Data Model Requirements (`domain_models/config.py` & `workflow.py`)
Extend Pydantic models to control the new workflow. Ensure massive arrays use `arbitrary_types_allowed=True` and generators (`Iterator`) to prevent OOM.

*   **`DistillationConfig`**: Configures Phase 1 (enable, `mace_model_path`, `uncertainty_threshold`, sampling counts).
*   **`ActiveLearningThresholds`**: Manages the two-tier evaluation (`threshold_call_dft`, `threshold_add_train`, `smooth_steps`).
*   **`CutoutConfig`**: Configures Phase 3 ($R_{core}$, $R_{buffer}$, `enable_pre_relaxation`, `enable_passivation`).
*   **`LoopStrategyConfig`**: Loop strategy settings (`use_tiered_oracle`, `incremental_update`, `replay_buffer_size`, `baseline_potential_type`).

---

## 5. Non-Functional & HPC Operations Requirements

### 5.1. State Management and Transactions
*   **Task-level Checkpointing**: Commit state to a local JSON/SQLite DB after every single DFT calculation or surrogate generation, not just per iteration. Allows resumability within seconds if an HPC job hits a Wall-time kill limit.
*   **Artifact Cleanup**: A parallel daemon process must automatically gzip or delete massive `.wfc` (wavefunction) and LAMMPS dump files immediately after successful learning/inference to prevent storage exhaustion.

### 5.2. Scheduler Integration
*   The `TieredOracle` must dispatch DFT calculations asynchronously using `concurrent.futures` across available nodes/GPUs. Subprocess calls to `PacemakerTrainer` must dynamically construct HPC prefixes (e.g., Slurm's `srun`).

---
End of Document
