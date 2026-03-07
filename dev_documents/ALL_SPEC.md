# PYACEMAKER Next-Generation Architecture Product Requirements Document (PRD)

**Version:** 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date:** 2026-02-28
**Status:** DRAFT

## 1. Project Background and Objectives

### 1.1. Limitations of the Current System (Phase 01) and Challenges
The basic orchestration of PYACEMAKER implemented in Phase 01 established the foundation for the Active Learning loop. However, critical physical and systemic limitations were discovered when scaling to practical HPC workloads (long-term Molecular Dynamics simulations involving tens of thousands to millions of atoms).

*   **Time-Continuity Break in MD:** When the simulation halts due to high uncertainty, retraining occurs, and the MD restarts from the initial structure. This prevents the simulation from reaching long-term physical phenomena like phase transformations or diffusion.
*   **Thermal Noise False Positives:** The system relies on a single uncertainty threshold to trigger a halt. It reacts overly sensitively to instantaneous spikes caused by thermal vibrations during MD (which are physically safe noise), leading to unnecessary calculations and infinite loops.
*   **Physical Failure due to Local Cutouts (Dangling Bond / Dipole Divergence):** Simply extracting an uncertain region (cluster) from a massive system into a vacuum and passing it to DFT creates numerous dangling bonds at the cut surfaces. This causes charge imbalance and dipole moment divergence. As a result, the DFT SCF loop fails to converge, or the model learns "garbage" (unphysical electronic states).
*   **Sluggish Batch Retraining & Catastrophic Forgetting:** The design of retraining the entire model from scratch using all data at every step causes an explosion in computational cost ($O(N)$). Furthermore, continuously adding only error structures degrades the predictive accuracy for stable bulk structures.
*   **LAMMPS Integration Fragility (System Fragility):** A crash in LAMMPS at the C++ layer (e.g., lost atoms) brings down the main Python process. The entire orchestrator halts without even saving its state.

### 1.2. Extraction of Best Practices from Prior Work (FLARE)
Based on an analysis of the FLARE architecture developed at Harvard University, we are introducing four paradigm shifts into this system.

1.  **Master-Slave Inversion (Inversion of Control):** Instead of Python controlling LAMMPS, Python is subordinated within the LAMMPS C++ loop (using `fix python/invoke` or periodic callbacks). This allows for "pause -> potential update -> resume" seamlessly without rewinding MD time.
2.  **Two-Tier Thresholds:** We separate the "threshold for calling DFT" (`threshold_call_dft`) from the "threshold for adding to training data" (`threshold_add_train`), providing resilience against thermal noise.
3.  **Global Calculation, Local Learning:** Electronic states are calculated in a physically stable environment, but only the forces on the central atoms (which had high uncertainty) are used to update the learning model.
4.  **Incremental Update:** Instead of batch retraining, we use delta learning (incremental updates) starting from previous weights, keeping computational costs at $O(1)$.

### 1.3. Fundamental Philosophy of the Next-Generation Architecture
Integrating the above challenges and the lessons from FLARE, we define a next-generation workflow centered around "Hierarchical Distillation," "Intelligent Cutout & Passivation," and **"Asynchronous Master-Slave MD (Seamless Resume)."**

To enable MD simulations of millions of atoms—which FLARE abandoned—we adopt an approach not of "not cutting out," but of "cutting out after perfect physical repair." Furthermore, by maximizing the use of distillation and fine-tuning from a foundation model (MACE) throughout the system, we aim to minimize the number of high-cost DFT calculation attempts (the biggest bottleneck) while achieving high-accuracy calculations approaching DFT in the target regions.

## 2. 4-Stage Hierarchical Distillation Workflow (Core Workflow)

The system sequentially executes the following four phases for the target system (e.g., a quaternary Fe-Pt-Mg-O system).

### Phase 1: Zero-Shot Distillation & Baseline Construction
**Objective:** To extract the "universal knowledge" (broad generalization capability) of foundation models like MACE-MP-0 without invoking DFT, and bake it into ACE.

1.  **Spatial Decomposition and Combinatorial Exploration:** Automatically define subsystems for all unary species and all binary pairs. For each subsystem, generate a structure pool covering random structures, strains, rattles, high-temperature snapshots, **variations in stoichiometry**, and the introduction of defects (vacancies, interstitials, antisites).
2.  **Information Maximization via DIRECT Sampling (Active Set Selection):** Utilize the existing `ActiveSetSelector` to extract the most information-dense and diverse structures (hundreds to thousands) from the massive structure pool. This eliminates redundant data and drastically reduces downstream inference and training costs.
3.  **Confidence Filtering:** Pass the extracted structures to `MACEManager` (the foundation model Oracle) to infer energy, forces, and uncertainty. Only structures with MACE uncertainty below the threshold ("safe structures MACE is confident in") are accepted as ground truth data.
4.  **Baseline ACE Training (LJ Delta Learning):** Launch `PacemakerTrainer` using the high-quality data that passed confidence filtering to learn a fundamental many-body potential (`base.yace`). To prevent atoms from passing through each other in close-proximity regions, apply Delta Learning from a Lennard-Jones (LJ) potential as the default configuration, ensuring a baseline for short-range repulsion.

### Phase 2: Validation & Stress Test
**Objective:** To verify that the baseline potential constructed in Phase 1 guarantees minimum physical stability in the production environment.

1.  **Physical Property Inspection of Parent Materials:** Launch the `Validator` to calculate the elastic constants (Born stability criteria), phonon dispersion (absence of imaginary frequencies), and equation of state (EOS) for the stable phases of each subsystem. If the passing criteria are not met, automatically increase the sampling density in Phase 1 and retrain.
2.  **Miniature MD Stress Test:** Create a scaled-down version of the production environment (e.g., a slab model of a few thousand atoms) and run MD with the constructed potential. Profile whether an early halt occurs or at which temperature range uncertainty increases (Uncertainty Map).

### Phase 3: Intelligent Cutout & Passivation
**Objective:** When an unknown local structure (interface, defect, collision) is encountered during large-scale MD and causes a halt, automatically generate a physically valid, clean cluster that can be computed by DFT.

1.  **Epicenter Identification based on Two-Tier Evaluation:** Trigger a halt only if the system's maximum uncertainty exceeds `threshold_call_dft` for several consecutive steps (filtering out thermal noise). Evaluate the individual atom uncertainty (Site-uncertainty), and identify the group of atoms exceeding `threshold_add_train` as the "epicenter."
2.  **Spherical Cutout and Weighting (Local Learning):** Utilize the existing `extract_local_region` to assign a `force_weight = 1.0` to atoms within radius $R_{core}$ from the epicenter, and `force_weight = 0.0` to atoms within radius $R_{buffer}$. Safely relocate the cutout cluster into a Periodic Boundary Condition (PBC) cell with a vacuum layer, focusing learning strictly on the core.
3.  **Pre-relaxation by MACE:** Extend the existing MLIP wrapper mechanism for foundation models (MACE). **Freezing the coordinates of the core atoms**, use MACE to minimize the energy (relax) of the coordinates of the atoms in the buffer region only. This resolves unnatural bond distortions caused by the cutout.
4.  **Auto-Passivation:** Automatically place dummy atoms like Fractional Hydrogen on broken bonds at the outer edge of the buffer region to neutralize the charge and dipole moment of the entire cluster.
5.  **Clean DFT Calculation:** Pass the physically and electrically stabilized cluster to the `QEDriver` and `DFTManager`. Fully utilize self-healing features (extending smearing or auto-adjusting mixing beta) to ensure SCF convergence and obtain the Ground Truth Force for the core atoms.

### Phase 4: Hierarchical Delta Learning
**Objective:** Use the small amount of precious DFT data obtained to sequentially update MACE and ACE, and then resume MD.

1.  **Finetune MACE (Awakening MACE):** Finetune the MACE model itself using the obtained DFT data. The foundation model fully understands the "specific interface physics of the target system" (Awakened MACE).
2.  **Explosive Generation of Surrogate Data:** Use the Awakened MACE as an Oracle to instantly generate and infer thousands of surrogate data points in the phase space around the halted region (random displacements or micro-MD).
3.  **Incremental Update of ACE:** Input the massive surrogate data and the anchor DFT ground truth data into `PacemakerTrainer` to update the ACE potential. Use Delta Learning from the previous potential and mix in a replay buffer to prevent catastrophic forgetting.
4.  **Master-Slave Resume:** Load the updated potential and safely resume MD from the exact step (time, coordinates, velocities) where it halted.

## 3. Module-Specific Requirements

### 3.1. `pyacemaker.utils.extraction` (Major Extension)
The module responsible for cluster extraction and passivation.

*   `extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms`
    *   **Input:** Massive ASE Atoms object, list of indices of target atoms exceeding `threshold_add_train`.
    *   **Processing:**
        *   Spherical extraction of $R_{core}$ and $R_{buffer}$ using `neighbor_list`.
        *   Assignment of `force_weight` array (Core=1.0, Buffer=0.0).
        *   `_pre_relax_buffer(cluster, mace_calc)`: Fix the core with `ase.constraints.FixAtoms` and relax the buffer using LBFGS with MACE.
        *   `_passivate_surface(cluster)`: Detect unbound dangling bonds and appropriately add H or pseudo-atoms.
    *   **Output:** Computable Atoms object with periodic boundaries, vacuum layer, and passivated edges.

### 3.2. `pyacemaker.core.oracle` (Multi-tiered)
Abstract the Oracle to transparently handle foundation models (MACE) and first-principles calculations (DFT).

*   `class MACEManager(BaseOracle)`: Wrapper for running inference like MACE-MP-0. GPU compatible. Must output uncertainty.
*   `class TieredOracle(BaseOracle)`: Manages the query strategy. Inferences with `MACEManager` first, routing to `QEDriver` (DFT) only if uncertainty exceeds a threshold.

### 3.3. `pyacemaker.core.engine` (LAMMPS Integration and Seamless Resume)
A robust engine that withstands LAMMPS crashes and continues time after a halt.

*   **Utilization of `fix python/invoke` (Recommended Approach):** Call the Python verification script every N steps directly from the LAMMPS C++ execution loop. If uncertainty exceeds the threshold, pause MD, run the Orchestrator in the background, dynamically reload the `pair_coeff`, and continue.
*   **Soft Start (Prevention of Temperature Spikes):** Automatically insert logic to apply a strong Langevin thermal bath (`fix langevin`) for the first few steps after resuming to thermalize the system and prevent energy discontinuity explosions.

### 3.4. `pyacemaker.core.trainer` (Pacemaker & MACE Finetune)
*   **FinetuneManager:** Wrapper to train the readout layer of the PyTorch MACE model for a short time using clean dataset from DFT.
*   **Incremental Update / Delta Learning in PacemakerTrainer:** Mix a fixed-size replay buffer randomly sampled from past training data (`training_history.extxyz`) into the current training set, carrying over previous weights to prevent catastrophic forgetting.

## 4. Data Model Requirements (`domain_models/config.py` & `domain_models/workflow.py`)

New workflow control parameters using Pydantic.

```python
class DistillationConfig(BaseModel):
    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="Safe MACE threshold")
    sampling_structures_per_system: int = 1000

class ActiveLearningThresholds(BaseModel):
    threshold_call_dft: float = Field(0.05, description="Halt criteria")
    threshold_add_train: float = Field(0.02, description="Target atom selection criteria")
    smooth_steps: int = Field(3, description="Consecutive steps for noise filtering")

class CutoutConfig(BaseModel):
    core_radius: float = Field(4.0, description="Force Weight 1.0 radius")
    buffer_radius: float = Field(3.0, description="Buffer layer radius")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

class LoopStrategyConfig(BaseModel):
    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, description="Replay buffer size")
    baseline_potential_type: str = Field("LJ", description="Base physics potential")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)
```

## 5. Non-Functional / HPC Operational Requirements

### 5.1. State Management and Transactions
*   **Task-level Checkpointing:** Commit state to a local JSON/SQLite database per DFT calculation and surrogate generation. Resumable within seconds if killed by HPC wall-time.
*   **Artifact Cleanup:** Background daemon process to automatically compress (gzip) or delete massive dump files and QE wavefunction files (`.wfc`) immediately after successful training/inference.

### 5.2. Scheduler Integration and Parallelization
*   Asynchronously dispatch Oracle (DFT) calculations to available nodes/GPUs using `concurrent.futures` or Dask.
*   Implement `JobDispatcher` to dynamically construct HPC prefixes (e.g., Slurm `srun`, PBS `mpiexec`).
