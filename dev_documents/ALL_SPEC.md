# PYACEMAKER Next-Generation Architecture Requirements Definition (PRD)

**Version:** 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date:** 2026-02-28
**Status:** DRAFT

## 1. Project Background and Objectives

### 1.1. Limitations and Challenges of the Current System (Phase 01)
The basic orchestration implemented in Phase 01 established the foundation for the Active Learning loop. However, when deployed to practical High-Performance Computing (HPC) scales (long-term molecular dynamics simulations involving tens of thousands to millions of atoms), the following fatal physical and systemic constraints became apparent:

*   **Time-Continuity Break:**
    When uncertainty exceeds the threshold and a halt is triggered, the MD simulation restarts from the initial structure after retraining. This design prevents reaching long-timescale physical phenomena such as phase transformations and diffusion.
*   **Thermal Noise False Positives:**
    Because a single uncertainty threshold determines the halt, the system reacts hypersensitively to instantaneous spikes caused by thermal vibrations during MD (which are physically safe noise), leading to unnecessary calculations and infinite loops.
*   **Physical Integrity Loss during Local Extraction (Dangling Bond / Dipole Divergence):**
    Simply cutting out uncertain regions (clusters) from a giant system into a vacuum and passing them to DFT causes massive dangling bonds at the cut surfaces. This creates charge imbalances and dipole moment divergences, resulting in DFT SCF loop non-convergence or the learning of "garbage" (non-physical electronic states).
*   **Sluggish Batch Retraining & Catastrophic Forgetting:**
    Retraining from scratch using all data at every step causes an explosion in computational cost ($O(N)$). Furthermore, continuously adding error structures degrades the description accuracy of stable bulk structures.
*   **LAMMPS Integration Fragility (System Fragility):**
    C++ layer crashes in LAMMPS (e.g., lost atoms) drag down the main Python process. The entire orchestrator halts without saving the state.

### 1.2. Best Practices Extracted from Prior Work (FLARE)
Based on codebase analysis of the FLARE architecture developed at Harvard University, we will introduce the following four paradigm shifts to our system:

1.  **Inversion of Control (Master-Slave Inversion):** Instead of Python controlling LAMMPS, Python is subordinated within the LAMMPS C++ loop (via `fix python/invoke` or periodic callbacks). This allows seamless "pause -> update potential -> resume" cycles without rewinding MD time.
2.  **Two-Tier Thresholds:** Separate the thresholds for calling DFT (`threshold_call_dft`) and for adding to the training data (`threshold_add_train`) to build resistance against noise.
3.  **Global Calculation, Local Learning:** Calculate electronic states in a physically stable environment, but use only the "forces of the central atoms" (which had high uncertainty) to update the learning model.
4.  **Incremental Update:** Use differential learning (Delta Learning/Incremental Update) starting from previous weights rather than batch retraining, keeping the computational cost at $O(1)$.

### 1.3. Core Philosophy of the Next-Generation Architecture
Integrating the above challenges and FLARE's lessons, we define the next-generation workflow centred around **"Hierarchical Distillation"**, **"Intelligent Cluster Extraction (Site-specific Cutout & Passivation)"**, and **"Asynchronous Master-Slave MD (Seamless Resume)"**.

To enable the "multi-million atom MD" that FLARE gave up on, we adopt the approach not of "avoiding cutouts," but of "cutting out after perfect physical restoration." Furthermore, by maximizing the use of distillation and fine-tuning from foundational models (MACE) throughout the system, we aim to minimize the number of expensive DFT calculations while achieving high-accuracy calculations approaching DFT in the target regions.

## 2. 4-Stage Hierarchical Distillation Workflow (Core Workflow)

The system will sequentially execute the following four phases for the target system (e.g., a quaternary Fe-Pt-Mg-O system):

### Phase 1: Zero-Shot Distillation & Baseline Construction
*   **Objective:** Extract the "universal common sense" (broad generalisation capability) of foundational models like MACE-MP-0 without invoking DFT, and bake it into ACE.
*   **Spatial Decomposition & Combinatorial Exploration:**
    Automatically define all elemental and binary subsystems from the input elements. Generate a massive structure pool covering random structures, strains, rattles, high-temperature snapshots, **stoichiometry variations**, and defects (vacancies, interstitials, antisites).
*   **Information Maximisation via DIRECT Sampling (Active Set Selection):**
    Utilise the existing `ActiveSetSelector` (DIRECT sampling / D-Optimality) to extract structures (hundreds to thousands) with the highest information density and diversity in feature space from the pool. This eliminates redundant data and minimises downstream inference/training costs.
*   **Confidence Filtering:**
    Pass the extracted structures to `MACEManager` (Foundational Model Oracle) to infer energy, forces, and uncertainty. Only adopt "safe structures where MACE is confident" (uncertainty below a threshold) as ground truth data.
*   **Baseline ACE Training (LJ Delta Learning):**
    Launch `PacemakerTrainer` using the high-quality data that covers a wide chemical space and passed the confidence filter. Train a basic many-body potential (`base.yace`). To prevent atoms from passing through each other at ultra-close ranges, apply Delta Learning from a Lennard-Jones (LJ) potential as the default configuration. This functionality should utilise existing classes to optimise parameters per element.

### Phase 2: Validation & Stress Test
*   **Objective:** Verify that the foundational potential built in Phase 1 ensures minimum physical stability in the production environment.
*   **Physical Property Inspection of Parent Materials:**
    Launch the `Validator` to calculate elastic constants (Born stability criteria), phonon dispersions (absence of imaginary frequencies), and equations of state (EOS) for stable phases of each subsystem (e.g., bcc-Fe, fcc-Pt, NaCl-MgO). If criteria are not met, automatically increase the sampling density of Phase 1 and retrain.
*   **Miniature MD Stress Testing:**
    Create a scaled-down production environment (e.g., slab models of several thousand atoms) and run MD. Profile early halts or temperature ranges where uncertainty rises (Uncertainty Map).

### Phase 3: Intelligent Cutout & Passivation
*   **Objective:** When an unknown local structure (interface, defect, collision) is encountered during large-scale MD and causes a halt, automatically generate a physically valid, clean cluster computable by DFT.
*   **Epicentre Identification based on Two-Tier Thresholds:**
    Trigger a halt only if the system's maximum MD uncertainty exceeds `threshold_call_dft` for several consecutive steps (filtering thermal noise). Evaluate individual atomic uncertainties using extended existing functions like `_get_max_gamma_atom_index`, and identify atoms exceeding `threshold_add_train` as the "epicentre."
*   **Spherical Cutout and Weighting (Local Learning):**
    Utilise existing `utils.extraction.extract_local_region` to assign `force_weight = 1.0` to atoms within radius $R_{core}$ and `force_weight = 0.0` within $R_{buffer}$ from the epicentre. Safely relocate the cutout cluster into a periodic boundary cell (PBC) with a vacuum layer using existing `utils.embedding.embed_cluster`.
*   **Boundary Pre-relaxation via MACE:**
    Extend the existing MLIP wrapper mechanism for the foundational model (MACE). Keep core atoms fixed (`Freeze`) and relax only the buffer region atoms using MACE to minimise energy. This eliminates unnatural bond distortions caused by the cutout.
*   **Auto-Passivation:**
    Automatically place dummy atoms like Fractional Hydrogen on broken bonds at the outer edge of the buffer region (especially for oxides like O and Mg) to neutralise the charge and dipole moment of the entire cluster (to be newly integrated into `utils.structure` etc.).
*   **Clean DFT Calculation:**
    Pass the physically and electrically stabilised cluster to the existing `QEDriver` and `DFTManager`. Fully utilise existing self-healing features (extending smearing or auto-adjusting mixing beta) to guarantee SCF convergence and acquire true forces for the core atoms.

### Phase 4: Hierarchical Delta Learning
*   **Objective:** Use the small amount of precious DFT data obtained to sequentially update MACE and ACE, then resume MD.
*   **MACE Awakening (Finetune MACE):**
    Fine-tune MACE itself using the obtained DFT data. The foundational model fully understands the "specific interface physics of the target system" (Awakened MACE).
*   **Explosive Generation of Surrogate Data:**
    Using Awakened MACE as an Oracle, instantly generate and infer thousands of surrogate data points in the phase space around the halt (random displacements, micro MD).
*   **ACE Delta Learning (Incremental Update):**
    Feed the massive surrogate data and anchor DFT ground truth data into `PacemakerTrainer` to update the ACE potential. To prevent computational explosion, do not train from scratch; perform differential learning from the previous potential and mix in the replay buffer.
*   **Seamless Resume (Master-Slave Resume):**
    Load the updated potential and safely resume MD from the exact step (time, coordinates, velocities) immediately following the halt.

## 3. Module Requirements

### 3.1. `pyacemaker.utils.extraction` (Major Extension)
The core module for cluster extraction and passivation.

*   `extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms`
    *   **Input:** Massive ASE `Atoms` object, index list of target atoms exceeding `threshold_add_train`.
    *   **Process:**
        *   Spherical extraction of $R_{core}$ and $R_{buffer}$ using `neighbor_list`.
        *   Assignment of `force_weight` arrays (Core=1.0, Buffer=0.0).
        *   `_pre_relax_buffer(cluster, mace_calc)`: Fix core with `ase.constraints.FixAtoms` and relax buffer using MACE via LBFGS.
        *   `_passivate_surface(cluster)`: Detect dangling bonds from electronegativity and bond radii, and append H or pseudo-atoms appropriately (added atoms have `force_weight`=0.0).
    *   **Output:** A computable `Atoms` object with periodic boundaries, an inserted vacuum layer, and applied passivation.

### 3.2. `pyacemaker.core.oracle` (Multi-tiered)
Abstract the Oracle to handle the foundational model (MACE) and First-Principles Calculation (DFT) transparently.

*   `class MACEManager(BaseOracle)`
    *   Wrapper executing inference for MACE-MP-0. GPU supported.
    *   Must output uncertainty based on ensemble variance or latent feature space distance, alongside energy and forces.
*   `class TieredOracle(BaseOracle)`
    *   Manages query strategy. When receiving a structure, infers with `MACEManager` first. Only falls back to `QEDriver` (DFT) if uncertainty exceeds a specific threshold.

### 3.3. `pyacemaker.core.engine` (LAMMPS Integration and Seamless Resume)
Robust engine surviving LAMMPS crashes and continuing time after halts. Applies FLARE's Master-Slave paradigm.

*   **`fix python/invoke` utilisation (Recommended Approach):**
    Directly call Python verification scripts every N steps from LAMMPS's C++ loop. If uncertainty exceeds the threshold, pause MD, run the Orchestrator (learning pipeline) in the background, dynamically reload `pair_coeff` upon completion, and continue MD.
*   **Process Isolation & `read_restart` (Fallback Approach):**
    If C++ integration poses technical hurdles, use a separate process. If LAMMPS crashes, the main loop survives. Fully inherit velocity distributions and ensemble states from periodically saved `.restart` files to resume.
*   **Soft Start (Temperature Spike Prevention):**
    To prevent system collapse due to energy discontinuities immediately after potential switching, automatically insert logic applying a strong Langevin thermal bath (`fix langevin`) for the first $N$ steps after resuming to thermalise the system.

### 3.4. `pyacemaker.core.trainer` (Pacemaker & MACE Finetune)

*   `FinetuneManager`:
    Wrapper to briefly train the near-final layers (Readout layer) of the MACE PyTorch model using clean datasets acquired from DFT.
*   **`PacemakerTrainer` Incremental Update / Delta Learning Enhancement:**
    To prevent batch learning computation explosion, carry over the previous potential state. Mix a fixed-size replay buffer randomly sampled from past training data (`training_history.extxyz`) into the current training set.
    *   Must guarantee functionality to auto-generate settings in `input.yaml` to execute Delta Learning from an LJ potential.

## 4. Data Model Requirements (`domain_models/workflow.py`)

Extensions to Pydantic models to control the new workflow.

```python
class DistillationConfig(BaseModel):
    """Phase 1: Zero-Shot Distillation Settings"""
    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="Safe threshold for MACE")
    sampling_structures_per_system: int = 1000

class ActiveLearningThresholds(BaseModel):
    """Two-Tier Thresholds inspired by FLARE"""
    threshold_call_dft: float = Field(0.05, description="Criteria to halt MD and call DFT")
    threshold_add_train: float = Field(0.02, description="Criteria to select atoms for training set")
    smooth_steps: int = Field(3, description="Consecutive steps threshold exceedance required to eliminate thermal noise")

class CutoutConfig(BaseModel):
    """Phase 3: Intelligent Cutout Settings"""
    core_radius: float = Field(4.0, description="Radius for Force Weight 1.0")
    buffer_radius: float = Field(3.0, description="Thickness of additional relaxation buffer layer")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

class LoopStrategyConfig(BaseModel):
    """Active Learning Loop Strategy Settings"""
    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, description="Number of past data points to retain to prevent catastrophic forgetting")
    baseline_potential_type: str = Field("LJ", description="Physical baseline potential (e.g., LJ)")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)
```

## 5. Non-Functional / HPC Operational Requirements

### 5.1. State Management and Transactions (Robust Checkpointing)
*   **Task-level Checkpointing:**
    Commit state to a local JSON or SQLite database per DFT calculation or surrogate generation, not coarse iteration-level saving. If an HPC job is forcefully killed by Wall-time limits, it must be resumable within seconds upon resubmission.
*   **Artifact Cleanup:**
    Run parallel daemon processes to automatically compress (gzip) or delete massive dump files from multi-million step MDs or giant Quantum Espresso wave function files (`.wfc`) immediately after successful training/inference.

### 5.2. Scheduler Integration and Parallelisation (HPC Dispatch)
*   Asynchronously dispatch Oracle (DFT) computations to available nodes/GPUs using `concurrent.futures` or Dask, avoiding serial execution.
*   Implement a job template function (`JobDispatcher`) to dynamically construct HPC environment prefixes (e.g., Slurm's `srun`, PBS's `mpiexec`) from environment variables when launching `PacemakerTrainer` subprocesses.

## 6. Proposed Implementation Phases (Milestones)

*   **Cycle 01: Core Extraction & Pre-relaxation Setup**
    Redesign `extraction.py`. Implement MACE-based pre-relaxation and H-passivation algorithms.
*   **Cycle 02: Master-Slave Inversion & Two-Tier Evaluator**
    Implement Two-Tier Threshold logic. Refactor LAMMPS engine for Master-Slave integration (via `read_restart` or `fix python/invoke`) with soft-start Langevin damping.
*   **Cycle 03: MACE Oracle Integration & Hierarchical Distillation Loop**
    Integrate `MACEManager` and `TieredOracle`. Implement Phase 1 Zero-Shot Distillation workflow.
*   **Cycle 04: Incremental Update (Delta Learning) & Seamless Resume**
    Extend trainer for Delta Learning from LJ potentials. Implement Replay Buffer management to prevent catastrophic forgetting. Connect trainer back to Orchestrator for seamless MD resume.
*   **Cycle 05: HPC Scaling & Robustness (Checkpointing)**
    Implement fine-grained task-level checkpointing and automated artifact cleanup daemon.
