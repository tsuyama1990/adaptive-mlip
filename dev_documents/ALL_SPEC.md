# PYACEMAKER NextGen Architecture Specification (PRD)

**Version:** 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date:** 2026-02-28
**Status:** DRAFT

## 1. Project Background and Purpose

### 1.1. Limitations of the Current System (Phase 01)
The basic orchestration implemented in Phase 01 established the foundation for the Active Learning loop. However, when deployed at practical High-Performance Computing (HPC) scales (long-timescale molecular dynamics simulations involving tens of thousands to millions of atoms), several critical physical and systemic limitations were discovered:

*   **Time-Continuity Break:** When the uncertainty exceeds the threshold and the simulation halts, the molecular dynamics (MD) restarts from the initial structure after the potential is retrained. This prevents the simulation from observing long-timescale physical phenomena like phase transformations or diffusion.
*   **Thermal Noise False Positives:** The system relies on a single uncertainty threshold to trigger a halt. This makes it overly sensitive to instantaneous spikes caused by thermal vibrations (which are physically safe noise), leading to unnecessary calculations and infinite loops.
*   **Physical Divergence from Local Cutouts (Dangling Bonds):** Extracting highly uncertain regions (clusters) from a massive system directly into a vacuum creates numerous dangling bonds at the cut surfaces. This causes charge imbalances and dipole moment divergence, which either prevents the Density Functional Theory (DFT) self-consistent field (SCF) loop from converging or forces the model to learn unphysical "garbage" electronic states.
*   **Catastrophic Forgetting and Batch Retraining Overhead:** Re-running batch training on the entire dataset at every step causes computational complexity to explode. Furthermore, constantly adding highly distorted error structures degrades the model's accuracy on stable bulk structures.
*   **System Fragility with LAMMPS Integration:** Crashes in the underlying C++ LAMMPS layer (such as "Lost atoms" errors) often terminate the main Python process. The entire orchestrator crashes without even saving the state.

### 1.2. Best Practices from Prior Art (FLARE)
Based on an analysis of the FLARE architecture (developed at Harvard University), we are introducing four paradigm shifts into the PyAceMaker system:

*   **Master-Slave Inversion:** Instead of having Python externally drive LAMMPS, we embed Python within the LAMMPS C++ execution loop (using `fix python/invoke` or regular callbacks). This allows the system to pause, update the potential, and seamlessly resume without rewinding the simulation time.
*   **Two-Tier Thresholds:** We decouple the uncertainty evaluation into a `threshold_call_dft` (for triggering a DFT calculation) and a `threshold_add_train` (for identifying which atoms to add to the training data). This significantly improves resilience against thermal noise.
*   **Global Calculation, Local Learning:** We ensure the environment remains physically stable before performing electronic state calculations, and only the forces on the central uncertain atoms are used to update the learning model.
*   **Incremental Update:** We move away from batch retraining. Instead, we use delta learning (incremental updates) starting from previous weights, keeping computational costs at a constant O(1).

### 1.3. NextGen Architecture Philosophy
Integrating the above solutions, we define a NextGen workflow centered around **Hierarchical Distillation**, **Intelligent Cluster Extraction (Site-specific Cutout & Passivation)**, and **Asynchronous Master-Slave MD (Seamless Resume)**.

To enable million-atom MD (which FLARE abandoned), we adopt the approach of "perfect physical repair before cutout." Furthermore, by maximizing the distillation and fine-tuning of foundation models (like MACE) across the system, we aim to minimize expensive DFT calls while achieving near-DFT accuracy in targeted regions.

---

## 2. Four-Stage Hierarchical Distillation Workflow

The system executes four sequential phases for a target material system (e.g., a 4-element system like Fe-Pt-Mg-O).

### Phase 1: Zero-Shot Distillation & Baseline Construction
**Purpose:** Extract generalized knowledge from foundation models (like MACE-MP-0) to create a baseline ACE potential without calling DFT.

1.  **Spatial Decomposition & Combinatorial Exploration:** The system automatically defines all single-element and binary sub-systems from the input elements. For each sub-system, it generates a massive pool of structures including random configurations, applied strain, rattled atoms, high-temperature snapshots, diverse stoichiometry variations, and defects (vacancies, interstitials, antisites).
2.  **Information Maximization via DIRECT Sampling:** Using the existing `ActiveSetSelector` (D-Optimality), the system selects a subset of structures (hundreds to thousands) that maximize information density and diversity in the feature space.
3.  **Confidence Filtering:** The selected structures are passed to the `MACEManager` (the Foundation Model Oracle) to infer energies, forces, and uncertainties. Only structures where MACE has high confidence (uncertainty below a threshold) are accepted as ground truth data.
4.  **Baseline ACE Training (LJ Delta Learning):** Using the filtered high-quality data, the `PacemakerTrainer` trains a foundational many-body potential (`base.yace`). To prevent unphysical atomic overlap at short distances, this training utilizes Delta Learning against a Lennard-Jones (LJ) baseline potential, optimizing parameters per element.

### Phase 2: Validation & Stress Testing
**Purpose:** Verify that the baseline potential from Phase 1 meets minimum physical stability requirements for production.

1.  **Physical Property Inspection:** The `Validator` calculates elastic constants, phonon dispersions, and equations of state (EOS) for stable phases of each sub-system (e.g., bcc-Fe, fcc-Pt, NaCl-MgO). If criteria (like the absence of imaginary phonon modes) are not met, the system automatically increases Phase 1 sampling density and retrains.
2.  **Miniature MD Stress Test:** The system runs a small-scale MD simulation (e.g., a few thousand atoms in a slab model). It profiles the temperatures at which uncertainty rises (generating an Uncertainty Map) or checks if early halts occur.

### Phase 3: Intelligent Cutout & Passivation
**Purpose:** Automatically generate a clean, physically valid cluster for DFT calculation when the massive MD simulation halts due to unknown local structures.

1.  **Epicenter Identification (Two-Tier Evaluation):** The MD simulation halts only if the maximum system uncertainty exceeds `threshold_call_dft` for a consecutive number of steps (ignoring noise). Then, the system evaluates individual atomic site uncertainties. Atoms exceeding `threshold_add_train` are designated as the "epicenter."
2.  **Spherical Cutout and Weighting:** Using existing extraction utilities, the system cuts out a core region (radius $R_1$, `force_weight=1.0`) and a buffer region (radius $R_2$, `force_weight=0.0`) around the epicenter. This isolates the learning target to the core.
3.  **Pre-relaxation via MACE:** A new wrapper allows the foundation model (MACE) to relax the atomic coordinates of the buffer region while strictly **freezing the core atoms**. This resolves unnatural bond distortions caused by the cutout.
4.  **Auto-Passivation:** Dangling bonds at the outer edge of the buffer (especially for electronegative elements like O or Mg) are automatically passivated using dummy atoms (like fractional hydrogen) to neutralize the cluster's charge and dipole moment.
5.  **Clean DFT Calculation:** The stabilized cluster is sent to the `DFTManager`. Using existing self-healing features (automatic smearing and mixing beta adjustments), the system ensures SCF convergence to obtain the true ground state forces for the core atoms.

### Phase 4: Hierarchical Delta Learning
**Purpose:** Use the rare, high-value DFT data to sequentially update MACE and ACE, and then resume the MD simulation.

1.  **MACE Awakening (Fine-tuning):** The acquired DFT data is used to fine-tune the MACE model, making it perfectly aware of the specific interfacial physics of the target system.
2.  **Explosive Surrogate Generation:** The "awakened" MACE acts as an Oracle to instantly generate and infer thousands of surrogate data points in the phase space surrounding the halt event (using random displacements).
3.  **ACE Incremental Update:** The `PacemakerTrainer` updates the ACE potential using the surrogate data, the true DFT anchor data, and a replay buffer of historical data. Crucially, it uses delta learning from the previous potential rather than training from scratch.
4.  **Seamless Resume:** The updated potential is loaded, and the MD simulation safely resumes from the exact time step, coordinates, and velocities it had when it halted.

---

## 3. Module Requirements Specification

### 3.1. `pyacemaker.utils.extraction`
This module manages cluster extraction and passivation.

*   `extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms`:
    *   **Input:** Massive ASE Atoms object, indices of atoms exceeding `threshold_add_train`.
    *   **Process:** Performs spherical extraction ($R_1$, $R_2$) via neighbor lists. Assigns `force_weight` arrays. Uses `_pre_relax_buffer` to freeze the core and relax the buffer with MACE. Uses `_passivate_surface` to detect dangling bonds and add dummy atoms.
    *   **Output:** An `Atoms` object ready for DFT calculation, featuring periodic boundaries, vacuum padding, and passivated surfaces.

### 3.2. `pyacemaker.core.oracle`
Abstracts the Oracle to handle both foundation models and DFT calculations.

*   `MACEManager(BaseOracle)`: Wrapper for MACE-MP-0 inference (GPU supported). Must output energies, forces, and uncertainties (based on ensemble variance or latent space distance).
*   `TieredOracle(BaseOracle)`: Manages routing logic. It first queries `MACEManager`. If the uncertainty exceeds a defined threshold, it falls back to the `DFTManager`.

### 3.3. `pyacemaker.core.engine`
A robust engine that withstands LAMMPS crashes and enables seamless resumption.

*   **Master-Slave Execution:** Utilizes LAMMPS `fix python/invoke` to call Python verification scripts from the C++ loop. Pauses MD, runs the orchestrator in the background, reloads `pair_coeff`, and resumes. Alternatively, relies on robust `.restart` file management with process isolation if C++ coupling fails.
*   **Soft Start:** To prevent structural explosion from energy discontinuities after a potential update, a strong Langevin thermostat is automatically applied for the first few steps upon resumption to re-thermalize the system.

### 3.4. `pyacemaker.core.trainer`
*   `FinetuneManager`: A wrapper to briefly fine-tune the readout layers of the MACE PyTorch model using clean DFT datasets.
*   `PacemakerTrainer`: Upgraded for incremental delta learning. It mixes a fixed-size replay buffer from historical data to prevent catastrophic forgetting. It automatically generates `input.yaml` files configured for LJ Delta Learning.

---

## 4. Data Model Requirements (`domain_models/workflow.py`)

New Pydantic models to control the advanced workflow.

```python
class DistillationConfig(BaseModel):
    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="Threshold for MACE confidence")
    sampling_structures_per_system: int = 1000

class ActiveLearningThresholds(BaseModel):
    threshold_call_dft: float = Field(0.05, description="Threshold to halt MD and call DFT")
    threshold_add_train: float = Field(0.02, description="Threshold to add atoms to training set")
    smooth_steps: int = Field(3, description="Consecutive steps required to exclude thermal noise")

class CutoutConfig(BaseModel):
    core_radius: float = Field(4.0, description="Radius for Force Weight 1.0")
    buffer_radius: float = Field(3.0, description="Thickness of relaxation buffer layer")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

class LoopStrategyConfig(BaseModel):
    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, description="Historical data size to prevent forgetting")
    baseline_potential_type: str = Field("LJ", description="Baseline physics potential type")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)
```

---

## 5. Non-Functional Requirements (HPC Operations)

### 5.1. Robust Checkpointing
*   **Task-level Checkpointing:** State is committed to a local SQLite/JSON database after every micro-task (e.g., one DFT calculation, one surrogate generation). This allows resumption within seconds if an HPC job is killed by wall-time limits.
*   **Artifact Cleanup:** Massive files (like QE `.wfc` files or MD dumps) are automatically compressed or deleted by a background daemon immediately after successful training/inference.

### 5.2. Scheduler Integration (HPC Dispatch)
*   The Oracle dispatches DFT calculations asynchronously to available nodes/GPUs using `concurrent.futures` or `Dask`.
*   A `JobDispatcher` dynamically assembles execution prefixes (e.g., Slurm's `srun`, PBS's `mpiexec`) from environment variables when launching `PacemakerTrainer` subprocesses.
