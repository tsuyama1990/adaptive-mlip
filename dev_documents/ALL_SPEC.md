# PYACEMAKER Next-Generation Architecture Requirements Specification (PRD)

**Version:** 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date:** 2026-02-28
**Status:** DRAFT

## 1. Project Background and Purpose

### 1.1. Limitations of the Current System (Phase 01) and Challenges
The basic orchestration of PYACEMAKER implemented in Phase 01 established the foundation of the Active Learning loop. However, critical physical and systemic limitations were discovered when anticipating deployment to practical HPC scales (long-term molecular dynamics simulations with tens of thousands to millions of atoms).

*   **Break in MD Continuity (Time-Continuity Break):**
    When the simulation halts because uncertainty ($\gamma$) exceeds the threshold, the MD restarts from the initial structure after retraining. This design prevents reaching long-term physical phenomena such as phase transformation and diffusion.
*   **Thermal Noise False Positives:**
    Halting is judged by a single uncertainty threshold. This causes the system to overreact to momentary spikes caused by thermal vibrations during MD (which is physically safe noise), leading to unnecessary calculations and infinite loops.
*   **Physical Breakdown from Local Cutouts (Dangling Bond / Dipole Divergence):**
    When an uncertain region (cluster) is simply cut out of a massive system and passed to DFT in a vacuum, a large number of dangling bonds are generated on the cut surface. This causes charge imbalance and dipole moment divergence. Consequently, the DFT's SCF loop fails to converge or the system learns "garbage" (unphysical electronic states).
*   **Sluggish Batch Retraining and Catastrophic Forgetting:**
    The design of redoing batch learning using all data at every step causes an explosion in calculation cost ($O(N)$). Furthermore, continuously adding error structures deteriorates the description accuracy of stable bulk structures.
*   **Fragility of LAMMPS Integration:**
    A LAMMPS crash at the C++ layer (e.g., Lost atoms) brings down the main Python process. The entire orchestrator stops without saving its state.

### 1.2. Extracting Best Practices from Previous Research (FLARE)
Based on an analysis of the code base of the FLARE architecture developed at Harvard University, we are introducing the following four paradigm shifts into this system.

*   **Master-Slave Inversion (Inversion of Control):**
    Instead of Python controlling LAMMPS, Python is subordinated within the C++ loop of LAMMPS (via `fix python/invoke` or regular callbacks). This allows for a seamless transition of "pause -> update potential -> resume" without rewinding the time of the MD.
*   **Two-Tier Thresholds:**
    We separate the threshold for calling DFT (`threshold_call_dft`) from the threshold for adding data to the training set (`threshold_add_train`), providing tolerance to thermal noise.
*   **Global Calculation, Local Learning:**
    Electronic states are calculated in a physically stable environment. However, only the force of the "central atom," which had high uncertainty, is used to update the learning model.
*   **Incremental Update:**
    Instead of batch retraining, we use differential learning (Delta Learning/Incremental Update) using past weights as initial values. This keeps the calculation cost at $O(1)$.

### 1.3. Basic Philosophy of the Next-Generation Architecture
Integrating the above challenges and the lessons from FLARE, we define a next-generation workflow centred on **Hierarchical Distillation**, **Intelligent Cluster Extraction (Site-specific Cutout & Passivation)**, and **Seamless Master-Slave MD Resume**.

To enable MD for millions of atoms, which FLARE abandoned, we adopt an approach of "physically perfect repair and cutout." We aim to minimise the number of costly DFT calculation attempts by maximising the use of distillation and fine-tuning from foundation models (MACE) throughout the system, while achieving high-accuracy calculations approaching DFT in the target region.

---

## 2. The 4-Phase Hierarchical Distillation Workflow (Core Workflow)

This system sequentially executes the following four phases for a target system (e.g., Fe-Pt-Mg-O quaternary system).

### Phase 1: Zero-Shot Distillation & Baseline Construction
**Purpose:** Extract the broad generalisation performance of foundation models like MACE-MP-0 and imprint it onto ACE without calling DFT at all.

*   **Space Decomposition and Combinatorial Exploration:**
    Automatically define subsystem structures for all pure elements and binary systems from the input element group. For each subsystem, generate a structure pool covering random structures, strain, rattle, high-temperature snapshots, and diverse stoichiometry variations or defects.
*   **Information Maximisation via DIRECT Sampling (Active Set Selection):**
    Utilise the existing `ActiveSetSelector` to extract the structures with the highest information density and diversity (hundreds to thousands) in the feature space from the massive structure pool. This eliminates redundant data and minimises inference/learning costs.
*   **Confidence Filtering:**
    Pass the extracted structures to the `MACEManager` (foundation model Oracle) to infer energy, forces, and uncertainty. Only structures with uncertainty below the MACE threshold ("safe structures MACE is confident in") are adopted as ground truth data.
*   **Baseline ACE Learning (LJ Delta Learning):**
    Launch the `PacemakerTrainer` using the high-quality data filtered through confidence filtering. To prevent atoms from passing through each other in ultra-close regions, apply Delta Learning from the Lennard-Jones (LJ) potential as a default configuration to guarantee a baseline for short-range repulsion.

### Phase 2: Validation & Stress Test
**Purpose:** Verify whether the basic potential built in Phase 1 can guarantee minimum physical stability in the production environment.

*   **Physical Property Inspection of the Parent Material:**
    Launch the Validator to calculate elastic constants, phonon dispersion, and Equations of State (EOS) for the stable phase of each subsystem. If it fails, automatically increase the sampling density of Phase 1 and retrain.
*   **Stress Test via Miniature MD:**
    Create a scaled-down version of the production environment (e.g., a slab model of a few thousand atoms) and run an MD with the constructed potential. Profile whether early halts occur and at what temperature range uncertainty increases (Uncertainty Map).

### Phase 3: Intelligent Cutout & Passivation
**Purpose:** When a massive MD encounters an unknown local structure and halts, automatically generate a physically valid, clean cluster that can be calculated by DFT.

*   **Epicentre Identification based on Two-Tier Evaluation:**
    Trigger a halt only if the maximum uncertainty of the MD system exceeds `threshold_call_dft` for several consecutive steps (excluding thermal noise). Evaluate the site-uncertainty of individual atoms, identifying those exceeding `threshold_add_train` as the "epicentre."
*   **Spherical Cutout and Weighting (Local Learning):**
    Utilise the existing `extract_local_region` to assign `force_weight = 1.0` to atoms within the core radius and `0.0` to atoms within the buffer radius. Safely reposition the cut-out cluster in a Periodic Boundary Cell (PBC) with a vacuum layer using existing embedding tools.
*   **Pre-relaxation via MACE:**
    Extend the existing MLIP wrapper mechanism for MACE. Freeze the coordinates of the core atoms and relax only the buffer region atoms using MACE. This resolves unnatural bond distortions during the cutout.
*   **Auto-Passivation:**
    Automatically place dummy atoms (e.g., Fractional Hydrogen) for broken bonds at the outer edge of the buffer region to neutralise the charge and dipole moment of the entire cluster.
*   **Clean DFT Calculation:**
    Pass the physically and electrically stabilised cluster to the `QEDriver` and `DFTManager`. Fully utilise self-healing functions to ensure SCF convergence and acquire the true force for the core atoms.

### Phase 4: Hierarchical Delta Learning
**Purpose:** Use the small amount of valuable acquired DFT data to sequentially update MACE and ACE, and then resume the MD.

*   **MACE Awakening (Finetune MACE):**
    Fine-tune the MACE model itself using the acquired DFT data.
*   **Explosive Generation of Surrogate Data:**
    Use the awakened MACE as an Oracle to instantly generate and infer thousands of surrogate data points in the phase space around the halt.
*   **ACE Delta Learning (Incremental Update):**
    Input the massive surrogate data and anchor DFT true value data into the `PacemakerTrainer` to update the ACE potential. Use differential learning from the previous potential and mix in a replay buffer to prevent calculation cost explosions.
*   **Seamless Resume (Master-Slave Resume):**
    Load the updated potential and safely resume the MD from the exact step, coordinates, and velocity immediately after the halt.

---

## 3. Module Requirements Specification

### 3.1. `pyacemaker.utils.extraction`
The core module responsible for cluster extraction and passivation.

*   `extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms`
    *   **Input:** Massive ASE Atoms object, list of target atom indices exceeding `threshold_add_train`.
    *   **Process:**
        *   Spherical extraction using neighbour lists.
        *   Application of `force_weight` arrays.
        *   `_pre_relax_buffer`: Relax the buffer using MACE while fixing the core.
        *   `_passivate_surface`: Detect unbonded hands based on electronegativity and add dummy atoms.
    *   **Output:** A computable Atoms object with periodic boundaries, a vacuum layer, and passivation applied.

### 3.2. `pyacemaker.core.oracle`
Abstracts the Oracle to handle foundation models (MACE) and First-Principles Calculations (DFT) transparently.

*   `class MACEManager(BaseOracle)`
    *   Wrapper for executing MACE inferences. GPU supported.
    *   Must output energy, forces, and uncertainty.
*   `class TieredOracle(BaseOracle)`
    *   Manages query strategies. First infers with `MACEManager`, then falls back to `QEDriver` (DFT) only if uncertainty exceeds the threshold.

### 3.3. `pyacemaker.core.engine`
A robust engine that withstands LAMMPS crashes and continues time after a halt.

*   **Leveraging `fix python/invoke`:**
    *   Call Python verification scripts every N steps directly from the LAMMPS C++ execution loop. If uncertainty is high, pause MD, run the Orchestrator, dynamically reload the potential, and continue.
*   **Process Isolation and `read_restart` (Fallback):**
    *   Run as a separate process. Inherit velocity distribution and ensemble state entirely from periodically saved `.restart` files.
*   **Soft Start:**
    *   Automatically insert logic to apply a strong Langevin heat bath for a few steps immediately after resuming to thermalise the system, preventing breakdowns due to energy discontinuities.

### 3.4. `pyacemaker.core.trainer`
*   `FinetuneManager`:
    *   Wrapper to train the readout layer of the MACE PyTorch model using clean DFT datasets.
*   **Enhanced Incremental Updates in `PacemakerTrainer`:**
    *   Mix a fixed-size replay buffer randomly sampled from past training data into the current training set to prevent catastrophic forgetting.

---

## 4. Data Model Requirements (`domain_models/workflow.py`)

New Pydantic models must be added to control the workflow.

*   `DistillationConfig`: Settings for Phase 1. Includes MACE model path, uncertainty threshold, and sampling structures per system.
*   `ActiveLearningThresholds`: Two-tier thresholds (`threshold_call_dft`, `threshold_add_train`, and `smooth_steps`) inspired by FLARE.
*   `CutoutConfig`: Settings for Phase 3. Includes core radius, buffer radius, and toggles for pre-relaxation and passivation.
*   `LoopStrategyConfig`: Strategy settings for the Active Learning loop, linking to the tiered oracle and incremental update toggles.

---

## 5. Non-Functional Requirements & HPC Operational Requirements

### 5.1. Robust Checkpointing
*   **Task-level Checkpointing:** Commit the state to a local SQLite/JSON DB for every DFT calculation and surrogate generation. Allow resuming within seconds if an HPC job is killed by a wall-time limit.
*   **Artifact Cleanup:** Automatically compress or delete massive dump files or QE wavefunction (`.wfc`) files immediately after successful training/inference.

### 5.2. Scheduler Integration (HPC Dispatch)
*   Asynchronously dispatch Oracles (DFT calculations) to available nodes/GPUs using `concurrent.futures`.
*   Implement a `JobDispatcher` to dynamically assemble HPC environment prefixes (e.g., Slurm's `srun`, PBS's `mpiexec`).
