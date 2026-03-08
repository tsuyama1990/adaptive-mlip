# PYACEMAKER Next Generation Architecture Requirements Specification (PRD)

**Version:** 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
**Date:** 2026-02-28
**Status:** DRAFT

## 1. Project Background and Objectives

### 1.1. Limitations of the Current System (Phase 01)
The basic orchestration of PYACEMAKER implemented in Phase 01 successfully established the foundation of the Active Learning loop. However, when assuming deployment at a practical HPC scale (long-duration molecular dynamics simulations with tens of thousands to millions of atoms), the following fatal physical and systemic constraints became apparent:

*   **Loss of MD Continuity (Time-Continuity Break):** When the simulation halts because uncertainty ($\gamma$) exceeds the threshold, the MD is designed to restart from the initial structure after retraining. This prevents reaching long-term physical phenomena such as phase transformations and diffusion.
*   **Hypersensitivity to Thermal Noise (False Positives):** Halts are judged by a single uncertainty threshold. This causes the system to react too sensitively to momentary spikes caused by thermal vibrations (which are physically safe noise), leading to unnecessary calculations and infinite loops.
*   **Physical Breakdown from Local Cutouts (Dangling Bond / Dipole Divergence):** If an uncertain region (cluster) is simply cut out from a massive system into a vacuum and passed to DFT, a huge amount of dangling bonds occurs on the cut surface. This causes charge imbalance and dipole moment divergence. As a result, the DFT SCF loop either fails to converge or learns "garbage" (non-physical electronic states).
*   **Sluggish Batch Retraining and Catastrophic Forgetting:** The design of re-doing batch training using all data at every step causes an explosion in computational cost ($O(N)$). Moreover, adding mostly error structures degrades the description accuracy of stable bulk structures.
*   **Vulnerability in LAMMPS Integration (System Fragility):** A LAMMPS crash (e.g., lost atoms) in the C++ layer drags down the Python main process, causing the entire orchestrator to stop without even saving the state.

### 1.2. Extraction of Best Practices from Prior Research (FLARE)
Based on codebase analysis of the FLARE architecture developed at Harvard University and others, the following four paradigm shifts will be introduced into this system:

*   **Master-Slave Inversion (Inversion of Control):** Instead of Python operating LAMMPS, Python is subordinated inside the LAMMPS C++ loop (using `fix python/invoke` or periodic callbacks). This allows for a seamless "pause -> update potential -> resume" without rewinding MD time.
*   **Two-Tier Uncertainty Thresholds:** We separate the "threshold for calling DFT (`threshold_call_dft`)" and the "threshold for adding to training data (`threshold_add_train`)" to provide tolerance against thermal noise.
*   **Global Calculation, Local Learning:** The electronic state is calculated in a physically stable environment, but only the "forces of the central atoms," which had high uncertainty, are used to update the learning model.
*   **Incremental Update:** Instead of batch retraining, computational cost is kept at $O(1)$ through delta learning (incremental updates) using past weights as initial values.

### 1.3. Basic Philosophy of the Next Generation Architecture
Integrating the above challenges and the lessons from FLARE, we define a next-generation workflow centered around **"Hierarchical Distillation"**, **"Intelligent Cluster Extraction (Site-specific Cutout & Passivation)"**, and **"Asynchronous Master-Slave MD (Seamless Resume)"**.

To enable "MD of millions of atoms," which FLARE gave up on, we adopt an approach of "physically perfectly repairing and then cutting out" rather than "not cutting out." Furthermore, by maximizing the utilization of distillation and fine-tuning from a foundation model (MACE) throughout the system, we aim to achieve high-accuracy calculations approaching DFT in the target region while minimizing the number of expensive DFT calculation attempts, which is the biggest bottleneck.

---

## 2. Four-Stage Hierarchical Distillation Workflow (Core Workflow)

This system executes the following four phases sequentially for the target system (e.g., a 4-element system like Fe-Pt-Mg-O).

### Phase 1: Zero-Shot Distillation & Baseline Construction
**Objective:** To extract the "universal common sense (broad generalization performance)" of foundation models like MACE-MP-0 without calling DFT at all, and bake it into ACE.

1.  **Spatial Decomposition and Combinatorial Exploration:** From the input elements, automatically define subsystems for all single elements ($N$ types) and all binary systems (${}_NC_2$ types). For each subsystem, generate a structure pool that comprehensively covers stoichiometry variations, defects (vacancies, interstitials, antisites), random structures, strain, rattle, and high-temperature snapshots.
2.  **Information Maximization via DIRECT Sampling (Active Set Selection):** Utilize the existing `ActiveSetSelector` (DIRECT sampling / D-Optimality) asset to extract structures with the highest information density and diversity (hundreds to thousands) in the feature space from the massive generated structure pool. This eliminates redundant data and minimizes downstream inference/learning costs.
3.  **Confidence Filtering:** Pass the extracted structures to the `MACEManager` (foundation model Oracle) to infer energy, forces, and uncertainty. Only structures with MACE uncertainty below the threshold ("safe structures MACE is confident about") are adopted as ground truth data.
4.  **Baseline ACE Training (LJ Delta Learning):** Launch the `PacemakerTrainer` using the high-quality data that covers a wide chemical space and passed confidence filtering to train the basic many-body potential (`base.yace`). To prevent atoms from passing through each other in the ultra-close range (avoiding physical breakdown), Delta Learning from the Lennard-Jones (LJ) potential is applied as the default configuration to ensure a baseline for short-range repulsion.

### Phase 2: Validation & Stress Test
**Objective:** To verify whether the foundation potential built in Phase 1 can guarantee minimum physical stability in the production environment.

1.  **Physical Property Inspection of Parent Materials:** Launch the `Validator` to calculate elastic constants (Born stability criteria), phonon dispersion (absence of imaginary frequencies), and Equation of State (EOS) for the stable phases of each subsystem (e.g., bcc-Fe, fcc-Pt, NaCl-MgO). If the passing criteria are not met, automatically increase the sampling density of Phase 1 and retrain.
2.  **Miniature MD Stress Test:** Create a scaled-down version of the production environment (e.g., a slab model of a few thousand atoms) and run MD with the built potential. Profile whether early Halts occur and in which temperature zones uncertainty increases (Uncertainty Map).

### Phase 3: Intelligent Cutout & Passivation
**Objective:** When encountering an unknown local structure (interface, defect, collision) during massive MD and halting, automatically generate a physically reasonable clean cluster that can be calculated by DFT.

1.  **Epicenter Identification based on Two-Tier Thresholds:** Trigger a Halt only if the system's maximum uncertainty in MD exceeds `threshold_call_dft` for several consecutive steps (excluding thermal noise). Then, evaluate individual atomic uncertainty (Site-uncertainty) and identify atoms exceeding `threshold_add_train` as the "epicenter".
2.  **Spherical Cutout and Weighting (Local Learning):** Use the existing `extract_local_region` to assign `force_weight = 1.0` to atoms within radius $R_{\text{core}}$ from the epicenter, and `force_weight = 0.0` to atoms within radius $R_{\text{buffer}}$. Use existing embedding utilities to safely reposition the cut-out cluster into a periodic boundary cell (PBC) with a vacuum layer, narrowing the learning target to the core.
3.  **Boundary Pre-relaxation by MACE:** Extend the existing MLIP wrapper mechanism for the foundation model (MACE). While keeping the coordinates of the core atoms fixed (Freeze), use MACE to minimize the energy (Relax) of only the atoms in the buffer region. This resolves unnatural bond distortions caused during cutout.
4.  **Auto-Passivation:** Automatically place dummy atoms like Fractional Hydrogen on broken bonds at the outer edge of the buffer region (especially O and Mg in oxides) to neutralize the charge and dipole moment of the entire cluster.
5.  **Clean DFT Calculation:** Pass the physically and electrically stabilized cluster to the `QEDriver` and `DFTManager`. Fully utilize self-healing functions (smearing extension, automatic mixing beta adjustment) to ensure SCF convergence and obtain the Ground Truth Force for the core atoms.

### Phase 4: Hierarchical Fine-Tuning (Delta Learning)
**Objective:** Update MACE and ACE in a chain reaction using a small amount of precious DFT data obtained, and resume MD.

1.  **Awakening MACE (Finetune MACE):** Finetune MACE itself using the obtained DFT data. The foundation model becomes fully aware of the "specific interfacial physics of the target system" (Awakened MACE).
2.  **Explosive Generation of Surrogate Data:** Using the Awakened MACE as an Oracle, instantly generate and infer thousands of surrogate data in the phase space around the halt (random displacements, micro-MD).
3.  **ACE Incremental Update:** Input massive surrogate data and anchor DFT truth data into `PacemakerTrainer` to update the ACE potential. To prevent computational explosion, do not train from scratch but use incremental learning from the previous potential, and mix in a replay buffer.
4.  **Seamless Resume (Master-Slave Resume):** Load the updated potential and safely restart MD from the exact step (time, coordinates, velocities) immediately following the halt.

---

## 3. Module Requirements

### 3.1. `pyacemaker.utils.extraction` (Major Expansion)
The core module of this architecture responsible for cluster extraction and passivation.

*   `extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms`
    *   **Input:** Massive ASE Atoms object, list of target atom indices exceeding `threshold_add_train`.
    *   **Processing:**
        *   Spherical extraction using neighbor lists for core and buffer.
        *   Assignment of `force_weight` arrays (Core=1.0, Buffer=0.0).
        *   `_pre_relax_buffer`: Fix core with `ase.constraints.FixAtoms`, relax buffer using MACE via LBFGS.
        *   `_passivate_surface`: Detect unbound hands from electronegativity and bond radius, appropriately add H or pseudo-atoms (`force_weight=0.0`).
    *   **Output:** Calculable `Atoms` object with periodic boundaries, vacuum layer, and passivation.

### 3.2. `pyacemaker.core.oracle` (Multi-tiered)
Abstracts the Oracle, allowing transparent handling of foundation models (MACE) and First-Principles calculations (DFT).

*   `class MACEManager(BaseOracle)`: Wrapper executing MACE inferences. Must output energy, forces, and uncertainty.
*   `class TieredOracle(BaseOracle)`: Manages the query strategy. Inferences with `MACEManager` first; falls back to `QEDriver` (DFT) only if uncertainty exceeds specific thresholds.

### 3.3. `pyacemaker.core.engine` (LAMMPS Integration & Seamless Resume)
A robust engine that withstands LAMMPS crashes and keeps time continuous after halts. Applies the FLARE Master-Slave paradigm.

*   **Utilization of `fix python/invoke` (Recommended):** Directly call Python validation scripts every N steps from the LAMMPS C++ execution loop. Pause MD if uncertainty exceeds threshold, run Orchestrator in background, and dynamically reload `pair_coeff` to continue.
*   **Process Isolation and `read_restart` (Fallback):** Isolate as a separate process if C++ coupling is an issue. Survive LAMMPS crashes by fully inheriting velocity distribution and ensemble state from periodically saved `.restart` files.
*   **Soft Start (Temperature Spike Prevention):** Automatically insert logic to thermalize the system using a strong Langevin bath for the first few steps immediately after a potential switch.

### 3.4. `pyacemaker.core.trainer` (Pacemaker & MACE Finetune)
*   **`FinetuneManager`:** Wrapper to briefly train the final layers (Readout layer) of the MACE PyTorch model using clean DFT datasets.
*   **Incremental Update & Delta Learning Enhancement:** Mix a fixed-size replay buffer randomly sampled from past training data (`training_history.extxyz`) into the current training set, preserving the previous potential state. Automatically generate settings in `input.yaml` to execute Delta Learning from LJ potentials.

---

## 4. Data Model Requirements (`domain_models/workflow.py`)

New workflow control Pydantic models.

```python
class DistillationConfig(BaseModel):
    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="Threshold where MACE is confident")
    sampling_structures_per_system: int = 1000

class ActiveLearningThresholds(BaseModel):
    threshold_call_dft: float = Field(0.05, description="Criterion to halt MD and call DFT")
    threshold_add_train: float = Field(0.02, description="Criterion to select atoms to add to training set")
    smooth_steps: int = Field(3, description="Consecutive steps required to exceed threshold to exclude thermal noise")

class CutoutConfig(BaseModel):
    core_radius: float = Field(4.0, description="Radius for Force Weight 1.0")
    buffer_radius: float = Field(3.0, description="Thickness of additional relaxation buffer layer")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

class LoopStrategyConfig(BaseModel):
    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, description="Number of past data points to retain to prevent catastrophic forgetting")
    baseline_potential_type: str = Field("LJ", description="Baseline physical potential (e.g., LJ)")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)
```

---

## 5. Non-Functional & HPC Operational Requirements

### 5.1. State Management and Transactions (Robust Checkpointing)
*   **Task-level Checkpointing:** Commit states to a JSON or SQLite-based local DB per single DFT calculation or single surrogate generation, rather than coarse iterations. Enables resuming within seconds if an HPC job is forcefully killed by Wall-time limits.
*   **Artifact Cleanup:** Run parallel daemon processes to automatically compress (gzip) or delete massive dump files or QE wavefunction files (`.wfc`) immediately after successful learning/inference.

### 5.2. Scheduler Integration and Parallelization (HPC Dispatch)
*   Asynchronously dispatch Oracle (DFT calculations) to available nodes/GPUs using `concurrent.futures` or `Dask` rather than serial execution.
*   Implement a JobDispatcher that dynamically assembles HPC environment prefixes (Slurm's `srun`, PBS's `mpiexec`) from environment variables when calling `PacemakerTrainer` subprocesses.
