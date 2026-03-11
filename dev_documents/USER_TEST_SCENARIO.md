# USER ACCEPTANCE TEST SCENARIOS AND TUTORIAL PLAN

## 1. Test Scenarios

### Scenario ID: UAT-01 - Zero-Shot Distillation & Baseline Initialization
**Priority: High**

**Description:**
This scenario is meticulously designed to validate the "Phase 1: Zero-Shot Distillation" requirement of the NextGen architecture. The primary objective is to conclusively demonstrate that an end-user, regardless of their previous experience with complex ab-initio workflows, can effortlessly initialize a completely new project and generate a physically robust baseline Machine Learning Interatomic Potential (MLIP) strictly by utilizing a pre-trained Foundation Model (such as the MACE-MP-0 model). Crucially, this must be accomplished without triggering a single, computationally expensive Density Functional Theory (DFT) calculation on the backend.

The user journey will begin by defining a relatively simple chemical system within the core configuration file, for example, a binary Mg-O (Magnesium Oxide) system. The user will then execute the initial command-line orchestration instruction (`pyacemaker --config config.yaml`). Upon execution of this command, the user must observe the system automatically, and without further prompting, generating a massive combinatorial pool of candidate atomic structures. This pool must encompass various stoichiometries, extreme lattice strains, and complex localized rattling perturbations to simulate finite temperature effects.

The critical verification phase occurs when the user observes the system routing these generated candidate structures directly through the `MACEManager` (acting as the Foundation Model oracle). The system must demonstrably filter this vast pool based strictly on the Foundation Model's internal, multi-dimensional uncertainty metrics, cleanly rejecting any complex structure where the deep learning model is not highly confident in its own prediction.

The user will then verify, by tailing the terminal output, that the accepted, high-confidence structures are directly passed to the `PacemakerTrainer`. This trainer must be seen fitting the initial baseline ACE potential (e.g., actively employing Lennard-Jones Delta Learning techniques to ensure correct, physically sound short-range repulsive behavior).

The ultimate verification step requires the user to manually inspect the generated `.log` files residing in the project directory, as well as the initial `training_history.extxyz` database. The user must explicitly confirm that the baseline potential artifact (e.g., `base.yace`) was successfully and cleanly created. Most importantly, the user must verify that the logging system explicitly states that the total number of actual DFT calculation calls made during this entire complex initialization phase was exactly zero. This specific scenario proves the ultimate efficiency of the new architecture, demonstrating a near-instantaneous cold-start capability by exclusively leveraging the generalized "universal chemistry" knowledge embedded within the chosen Foundation Model. The entire process must be entirely automated, requiring absolutely zero manual intervention, parameter tweaking, or script modification after the initial YAML configuration is provided, successfully delivering a true "magic button" experience for new scientific users.

### Scenario ID: UAT-02 - Intelligent Cutout & Master-Slave MD Simulation
**Priority: High**

**Description:**
This highly complex scenario is designed to rigorously validate the core physical execution paradigms introduced in Phase 3 of the architecture: specifically, the "Master-Slave Inversion" control flow and the sophisticated "Intelligent Cutout" mathematical mechanisms. The overarching goal is to definitively prove that the system can run a continuous, massively parallelized molecular dynamics simulation, autonomously detect a highly specific physical anomaly, extract a physically sound and neutral atomic cluster without causing a fatal segmentation fault, and subsequently resume the simulation flawlessly.

The user will begin by initiating an MD simulation utilizing the previously generated baseline potential. This simulation should be run on a slightly larger test system (e.g., a 1000-atom supercell to provide enough bulk context). To reliably and predictably trigger the active learning logic, the user will introduce a deliberate, high-energy "defect" (such as a multi-atom vacancy, an anti-site defect, or a highly strained local interfacial region) directly into the initial structure file. This action forces the local atomic uncertainty to rapidly rise above the strictly configured `threshold_call_dft` limit.

The user must then actively monitor the real-time simulation output. They must explicitly observe the newly implemented `TwoTierEvaluator` detecting the engineered anomaly. The evaluator must then pause the massive MD simulation cleanly, utilizing the `fix python/invoke` mechanism, without causing the underlying LAMMPS C++ process to crash, hang, or lose critical atomic phase-space data. The terminal logs must show the system correctly and algorithmically identifying the specific "epicenter" atoms—defined strictly as those atoms whose individual uncertainty metric exceeds the lower `threshold_add_train` limit.

Following the successful halt, the user will inspect the detailed extraction logs to confirm the execution of the `extract_intelligent_cluster` Python function. They must mathematically verify that the system successfully isolated a perfectly spherical sub-cluster based exclusively on the configured core and buffer radii parameters. Furthermore, the extensive logs must explicitly indicate that the buffer region was dynamically pre-relaxed using the Foundation Model's optimizer while the critical core atoms remained absolutely frozen via ASE constraints. The logs must also prove that any surface dangling bonds created by the spherical cutout were automatically detected and chemically passivated (e.g., capped with fractional Hydrogen atoms to maintain charge neutrality).

Finally, the user must observe the system successfully routing this mathematically clean cluster to the simulated DFT solver, completing the high-speed surrogate data generation using the awakened MACE model, performing the rapid incremental training update via Pacemaker, and then—crucially—seamlessly resuming the paused MD simulation. The resume process must provably start from the exact time step, atomic coordinates, and atomic velocities recorded at the precise moment of the halt. Furthermore, the user must verify the application of the "Soft Start" Langevin thermostat in the initial resume steps to prevent unphysical energy spikes, demonstrating the ultimate robustness of the continuous execution paradigm.

### Scenario ID: UAT-03 - Incremental Delta Learning (O(1) Scalability)
**Priority: Medium**

**Description:**
This critical scenario rigorously tests the "Phase 4: Hierarchical Delta Learning" subsystem. The primary, uncompromising focus is proving that the system successfully and permanently prevents the phenomenon of catastrophic forgetting while simultaneously maintaining strict O(1) computational scalability during the machine learning training phase, regardless of exactly how long the simulation has been running or exactly how much historical data has been accumulated in the filesystem.

The user will configure a deliberately prolonged MD simulation explicitly designed to encounter multiple, highly distinct physical anomalies (e.g., sequential collision cascades), forcing the system to halt and execute the retraining pipeline several times sequentially. The user will allow the simulation to proceed naturally through the first halt event, observing the standard extraction, DFT calculation, and the initial, baseline training update.

The critical validation metrics are gathered during the second (and all subsequent) halt events. When the second distinct anomaly is detected by the evaluator, the user must carefully and meticulously scrutinize the outputted training logs and the recorded performance timing metrics. The user must strictly verify that the `PacemakerTrainer` binary does *not* attempt to load the entire accumulated history file (`training_history.extxyz`) into system RAM, nor does it attempt to fit a new ACE potential completely from scratch.

Instead, the user must see explicit, undeniable evidence of the "Incremental Update" strategy being engaged. The diagnostic logs must explicitly confirm that the underlying trainer initialized the tensor optimization routines using the precise mathematical weights of the potential generated during the *first* halt event. Furthermore, the user must verify via the logs that the training dataset constructed for this specific second update consists solely of three elements: the newly acquired anchor DFT data, the rapidly generated localized surrogate data, and a strictly size-limited, randomly sampled subset (the replay buffer) drawn uniformly from the historical data file.

To definitively prove the O(1) computational scalability requirement, the user will capture and compare the wall-clock execution time of the active training phase during the second halt directly against the recorded training time of the first halt. The user must verify mathematically that these two measured durations are roughly equivalent (falling well within an acceptable statistical variance of 5%). This comparison definitively demonstrates that the computational cost of updating the machine learning model remains strictly bounded, predictable, and constant, completely independent of the total volume of historical data accumulated over the entire lifetime of the advanced active learning simulation.

---

## 2. Behavior Definitions (Gherkin)

The following Gherkin scenarios define the exact, uncompromising expected behavior of the system under various critical conditions. These definitions serve as the strict executable specifications for both automated end-to-end testing pipelines and manual user acceptance verification.

**Feature: Zero-Shot Distillation Pipeline Initialization**

```gherkin
  Scenario: Successfully initializing a baseline potential without invoking any ab-initio DFT calculations
    Given a new, empty project directory is established on the filesystem
    And the project is correctly configured for the "Mg-O" binary chemical system within the main YAML file
    And the configuration file (`config.yaml`) defines a structurally valid `DistillationConfig` object
    And the `DistillationConfig` is explicitly and deliberately set to `enable: true`
    And the `mace_model_path` variable points to a locally available, pre-trained Foundation Model file (e.g., `mace-mp-0-medium.model`)
    And the `uncertainty_threshold` for the Foundation Model is strictly and numerically set to exactly 0.05 eV/A
    When the user executes the main terminal orchestrator command (`pyacemaker --config config.yaml`) to begin Iteration 0 (the Cold Start phase)
    Then the internal system should automatically generate a massive, highly diverse combinatorial structure pool encompassing various stoichiometries and physical strains
    And the `MACEManager` component should evaluate the entire generated pool and strictly filter out any atomic structures exhibiting an uncertainty value greater than 0.05 eV/A
    And the `PacemakerTrainer` component should exclusively utilize the filtered, high-confidence structures to train and successfully output an initial `base.yace` potential file to disk
    And the system's execution logs must explicitly and unambiguously state that the total number of calls made to the `DFTManager` during this entire iteration was exactly 0
```

**Feature: Two-Tier Evaluation, Noise Filtering, and Intelligent Cutout**

```gherkin
  Scenario: Accurately halting an MD simulation to extract a physically passivated, strain-free local cluster
    Given an active, massively parallelized Molecular Dynamics simulation is running continuously using the previously generated baseline ACE potential
    And the `ActiveLearningThresholds` are rigorously configured in the YAML file (specifically: `threshold_call_dft=0.05`, `threshold_add_train=0.02`, and `smooth_steps=3`)
    And the LAMMPS MD engine is periodically and successfully invoking the Python `TwoTierEvaluator` logic via synchronous C++ callbacks (e.g., utilizing `fix python/invoke`)
    When the calculated maximum atomic uncertainty within the entire massive simulation box exceeds 0.05 eV/A for 3 completely consecutive, uninterrupted simulation steps
    Then the `TwoTierEvaluator` should successfully trigger an `MDHaltInterrupt` exception
    And the exception should cause the running MD simulation to pause gracefully, writing a restart file, without ever terminating the parent Python orchestrator process
    And the system should iterate over the final recorded uncertainty array to identify all specific atomic indices with an uncertainty > 0.02 eV/A, designating them as the primary epicenter
    And the `extract_intelligent_cluster` Python function should successfully isolate a mathematically perfect spherical sub-cluster utilizing a `core_radius` of exactly 4.0 Angstroms and a `buffer_radius` of exactly 3.0 Angstroms
    And the atomic coordinates residing within the defined core region must be strictly and computationally frozen utilizing the `ase.constraints.FixAtoms` class
    And the atomic coordinates residing within the defined buffer region must be geometrically relaxed utilizing an LBFGS optimization routine driven strictly by the Foundation Model to relieve all boundary strain
    And any severed, dangling covalent bonds located on the exterior surface of the buffer region must be algorithmically detected and chemically passivated utilizing dummy Hydrogen atoms
```

**Feature: Seamless Simulation Resume and O(1) Incremental Delta Learning**

```gherkin
  Scenario: Executing an incremental potential update and flawlessly resuming the halted MD simulation
    Given the overall system has previously halted and successfully extracted a physical, fully passivated, electrically neutral cluster
    And the `DFTManager` has successfully completed the rigorous ab-initio SCF calculation and returned the absolute ground truth forces specifically for the frozen core atoms
    And the `FinetuneManager` has briefly awakened the underlying Foundation Model neural network utilizing this newly acquired ground truth data
    When the `PacemakerTrainer` initiates a targeted incremental update sequence utilizing the mathematical tensor weights of the previously active potential
    And the final training dataset is mathematically composed of the ground truth data, the MACE-generated local surrogate data, and a randomly, uniformly sampled historical replay buffer
    Then the underlying Fortran/C++ trainer should successfully converge and deploy the new, highly refined ACE potential file directly to the active working directory
    And the main Python Orchestrator should securely instruct the paused MD engine to immediately resume the physical simulation
    And the MD engine must provably restart from the exact, precise step number, atomic coordinates, and atomic velocities recorded in the binary restart file at the moment of the halt
    And the MD engine must automatically and dynamically apply a "Soft Start" strong Langevin thermostat for the first N resume timesteps to safely and smoothly thermalize any abrupt potential energy discontinuities
```

---

## 3. Tutorial Strategy

To provide an exceptional, frictionless user experience and ensure the highly complex NextGen architecture is easily verifiable by a wide range of researchers, external developers, and automated Continuous Integration (CI) pipelines, all User Acceptance Testing (UAT) scenarios and general introductory usage tutorials will be strictly consolidated into a single, highly interactive Marimo notebook.

### "Mock Mode" vs "Real Mode"

The central tutorial notebook will be architecturally designed with a globally available, easily toggleable "Mock Mode" configuration flag. This dual-mode approach is absolutely essential for accessibility and rapid verification.

- **Mock Mode (Default/CI Environment):** When this critical flag is enabled, the underlying tutorial seamlessly utilizes highly optimized, lightweight Fake Python implementations (e.g., substituting the heavy `MACEManager` with a `FakeMACECalculator`, replacing the licensed `QEDriver` with a `DummyDFTManager`, and bypassing the GPU-intensive `PacemakerTrainer` with a `FakePacemakerTrainer`). These Fakes perfectly simulate the complex computational physics and machine learning steps mathematically without doing the real work. This elegant design allows any user to run the entire, complex end-to-end active learning loop on a standard, low-power laptop in a matter of seconds. It focuses the user's attention entirely on verifying the high-level workflow logic, understanding the strict Pydantic configuration validation rules, observing the correct state transitions, and validating the robust behavior of the central Orchestrator. Crucially, this specific mode is also the mode utilized by the headless GitHub Actions CI pipeline to ensure the core Python code remains logically sound without requiring expensive, dedicated GPU runners or highly restricted, proprietary DFT software licenses on the build servers.
- **Real Mode:** When the toggle flag is explicitly and manually disabled by an advanced user, the tutorial immediately attempts to connect to actual, multi-gigabyte PyTorch Foundation Models loaded into VRAM, and attempts to execute real, MPI-parallelized LAMMPS and Quantum ESPRESSO C++ binaries installed on the host operating system. This resource-intensive mode is intended exclusively for final deployment verification on actual, high-performance HPC infrastructure, allowing dedicated researchers to definitively prove the overarching system works flawlessly with their specific, locally compiled binaries, network topologies, and highly specific hardware configurations.

---

## 4. Tutorial Plan

All specified UAT scenarios, comprehensive onboarding instructions, and deeply technical architectural explanations will be implemented and maintained within a **SINGLE**, easily executable Marimo Python file strictly located at:
`tutorials/UAT_AND_TUTORIAL.py`

This rigid, single-file approach ensures maximum simplicity, portability, and reproducibility for end-users. Users will never need to navigate complex, deeply nested directory structures, set confusing `PYTHONPATH` variables, or run multiple, isolated, fragile bash scripts to understand the system. The interactive notebook will logically contain the following highly structured execution cells:

1. **Introduction & Environment Setup:** This initial cell provides a high-level, easily digestible explanation of the complex Hierarchical Distillation architecture. It is immediately followed by the crucial UI toggle switch for selecting between "Mock Mode" and "Real Mode," and executes the necessary Python commands to dynamically initialize the required, isolated temporary directory structures.
2. **Configuration Definition & Validation:** This is a highly interactive, educational cell where the user explicitly defines the deeply nested `PyAceConfig` dictionaries in Python code. It is designed to specifically showcase, document, and explain the physical meaning behind the parameters within the new `DistillationConfig`, the geometric parameters of the `CutoutConfig`, and the delicate balance of the `ActiveLearningThresholds`. The cell will intentionally introduce a validation error, catch it, and explain why Pydantic rejected the configuration to demonstrate the system's safety guardrails.
3. **Execution of Phase 1 (Zero-Shot Initialization):** This cell acts as the primary trigger for the orchestrator, initiating iteration 0. The visual output generated by this cell will explicitly verify the rapid generation of the baseline `.yace` potential and definitively confirm via parsed logs that absolute zero ab-initio DFT calls were made during the process, completely satisfying the rigorous demands of Scenario UAT-01.
4. **Execution of Phases 2-4 (The Core Active Learning Loop):** This represents the most complex cell in the notebook, executing a simulated molecular dynamics step. It will programmatically inject a severe structural anomaly into the system to violently trigger the `TwoTierEvaluator` thresholds. It will visually output the rich extraction logs, clearly demonstrating the intelligent cutout spherical geometry and the chemical passivation process. Finally, it will mathematically verify the execution of the O(1) incremental training update and the seamless, continuous resume mechanics, simultaneously satisfying the highly demanding requirements of both Scenario UAT-02 and UAT-03.
5. **Physical Validation & Finalization:** The final execution cell will parse and elegantly display the generated physical validation report (e.g., rendering graphs of the simulated Born stability matrices and plotting the phonon dispersion checks), visually confirming for the user the absolute physical readiness and thermodynamic stability of the final generated machine learning potential.

---

## 5. Tutorial Validation

To absolutely ensure the tutorial remains functional, technically accurate, and tightly coupled to the underlying Python codebase as the core algorithms evolve over time, a strict validation protocol is enforced:

1. The `tutorials/UAT_AND_TUTORIAL.py` file is architecturally designed to be fully executable in completely headless, non-interactive environments (like Docker containers).
2. The Continuous Integration (CI) pipeline will be explicitly and permanently configured to execute the tutorial script strictly in "Mock Mode". It will bypass the Marimo UI and use the standard, unadorned Python interpreter (e.g., executing the command `uv run python tutorials/UAT_AND_TUTORIAL.py`).
3. The execution of this script must complete with a strict, non-negotiable `0` exit code. This rigid requirement definitively verifies that all complex Pydantic configurations load correctly, all Orchestrator state transitions execute in the mathematically proper sequence, and all mock data flows function seamlessly without raising unhandled exceptions, TypeErrors, or segmentation faults. Note that while the Marimo framework provides a beautiful interactive UI for humans, the `marimo test` or `marimo run` CLI commands are known to hang or timeout in headless, restricted Docker environments; therefore, direct standard Python execution is the absolute mandated method for achieving reliable CI validation.
