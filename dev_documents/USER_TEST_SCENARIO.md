# User Acceptance Test Scenarios for PyAceMaker

## 1. Test Scenarios

### Scenario ID: UAT-01
**Priority:** High
**Title:** Zero-Shot Baseline Distillation Success
**Description:**
This scenario tests the ability of the system to generate a fully functioning baseline machine learning interatomic potential without ever invoking an expensive density functional theory (DFT) calculation. The user configures a PyAceConfig yaml file specifying elements (e.g., Fe and O) and enables the `DistillationConfig`. The system must automatically generate a combinatorial pool of structures including isolated atoms, pure bulk phases, and heavily distorted mixed phases. It must then leverage the `ActiveSetSelector` to identify the most geometrically diverse subset. These structures are evaluated solely by the mocked MACE foundation model. Structures passing the uncertainty threshold are selected, and the `PacemakerTrainer` generates a baseline `base.yace` file coupled with a Lennard-Jones core. The test verifies that the resulting potential successfully passes validation checks for elasticity and phonon stability on the parent bulk structures.
To further elaborate on the expected behavior, the orchestrator should systematically synthesize ideal crystal lattices and meticulously traverse all quaternary and ternary elemental combinations requested in the yaml file. The active set selector's D-Optimality sampling algorithm is responsible for heavily pruning this vast configuration space, explicitly retaining the unique outliers that maximize the volumetric span of the geometric features while aggressively discarding redundant bulk-like coordinates. Following this, the MACE model will function as an autonomous oracle, predicting forces, energies, and a quantified uncertainty metric for each retained cluster. Any configuration possessing an uncertainty metric strictly greater than the defined parameter must be mercilessly filtered out of the training corpus. Ultimately, the system must utilize the remaining highly confident subset to train the ACE polynomial potential using the `PacemakerTrainer`. We must explicitly observe the generation of a valid `input.yaml` in the designated temporary training directory containing the proper Lennard-Jones repulsive core integration parameters before the `pace_train` subprocess is triggered. The final validation asserts that the `base.yace` file exists, possesses a non-zero byte size, and that invoking the Validator subsystem against the `bcc-Fe` and `rocksalt-MgO` parent structures returns elastic tensors and phonon dispersion frequencies that conform entirely to the physical stability criteria, thereby completing the zero-shot distillation phase entirely without expensive Quantum Espresso calculations.

### Scenario ID: UAT-02
**Priority:** High
**Title:** Thermal Noise Filtration via Two-Tier Evaluator
**Description:**
This scenario verifies that the simulation does not prematurely halt due to benign thermal vibrations. The system is configured with `ActiveLearningThresholds` setting a `smooth_steps` value of 5. A miniature molecular dynamics run is launched. During the run, the engine deliberately injects an artificial uncertainty spike that exceeds the `threshold_call_dft` limit for exactly 3 steps, before returning to a baseline safe level. The expectation is that the Two-Tier Evaluator registers the spike but continues the simulation because the duration did not meet the `smooth_steps` requirement. The MD must complete its designated timeframe without ever pausing or triggering the extraction and training pipeline, proving that the system has successfully ignored the noise.
This scenario is fundamentally critical for ensuring that high-temperature molecular dynamics simulations can progress uninterrupted across extensive timescales. Traditional orchestrators utilize a single, absolute threshold, making them hypersensitive to natural, momentary phase-space excursions that naturally subside within a few femtoseconds of simulation time. This test rigorously verifies the new stateful signal processing logic embedded within the `TwoTierEvaluator`. We will programmatically synthesize a stream of uncertainty values mimicking thousands of atomic timesteps. This synthetic stream will contain high-frequency Gaussian noise alongside carefully parameterized step-functions designed to cross the `threshold_call_dft` explicitly. The orchestrator must constantly evaluate this rolling window. When the artificial spike occurs at timestep $T=100$, the internal `HaltTrigger` boolean flag must strictly remain evaluated as `False`. The orchestrator should merely log a "Thermal Noise Detected, Ignoring" warning message to the standard output and immediately return execution control back to the `lammps` subprocess. Only if the stream of uncertainty values remains continuously elevated for a duration strictly greater than or equal to the defined `smooth_steps` parameter (in this case, 5 consecutive evaluations) should the system finally issue the formal halt command and serialize the molecular dynamics state into the designated `restart` binary format. By explicitly completing the predefined MD timeframe and generating the final `thermo` output logs without ever initiating the heavy, O(N^3) density functional theory extraction and delta-learning pipelines, the orchestrator definitively proves its immunity to benign thermal noise and false positives.

### Scenario ID: UAT-03
**Priority:** Critical
**Title:** Intelligent Cutout and Auto-Passivation on Anomaly Detection
**Description:**
This test ensures the safe, physical extraction of an unknown local environment. An MD simulation is run, and an anomaly (e.g., a massive defect collision) triggers a sustained high uncertainty, pausing the simulation. The system identifies the specific atom with the highest uncertainty. The `extract_intelligent_cluster` algorithm carves a sphere based on the configured radii. Crucially, the system must then correctly identify dangling bonds at the buffer boundary (e.g., an oxygen atom with only two neighbors instead of six) and append fractional hydrogen atoms via `_passivate_surface` to enforce neutrality. Finally, the buffer region is relaxed using the MACE oracle while the core remains frozen. The final structure is verified for electrical neutrality and lack of overlapping atomic radii before it is passed to the mocked DFT oracle.
The sheer complexity of this cluster manipulation is the primary defense against the infamous "Dangling Bond and Dipole Divergence" phenomenon that plagues naive active learning loops. When the orchestrator commands the extraction sequence, it first identifies the absolute index of the "epicenter" atom. The system then mathematically computes the Cartesian distances of all surrounding atoms, carving an inner core region defined by the `core_radius` scalar, and an outer buffer shell defined by the `buffer_radius` scalar. The test must programmatically assert that the core atoms receive an internal ASE `Atoms.info["force_weight"]` array assignment of exactly 1.0, while all other atoms receive an assignment of 0.0. Subsequently, the `_passivate_surface` method activates. We will strictly verify that the algorithm traverses the topology using the updated `neighbor_list`, correctly identifying under-coordinated species like Oxygen or Magnesium that lie precisely on the physical boundary of the cut surface. The orchestrator must dynamically generate fractional Hydrogen atoms or appropriate dummy species, appending them to the structure at physically sensible bond distances relative to the host atoms' covalent radii. Following this complex geometric manipulation, the test asserts that the `MACEManager` executes the pre-relaxation step. By applying `FixAtoms` constraints mapped explicitly to the indices of the inner core, the local optimization routines (e.g., LBFGS) must be shown to mutate the coordinates of the buffer and passivated surface atoms while leaving the epicenter coordinates utterly pristine. The final, overarching assertion demands that the resulting cluster object—now fully stabilized, passivated, and encased within a sufficient vacuum layer via periodic boundary conditions—produces an absolute dipole moment approaching zero, ensuring the Quantum Espresso driver receives an idealized, physically coherent structure capable of rapid electronic self-consistent field convergence.

### Scenario ID: UAT-04
**Priority:** Critical
**Title:** Seamless MD Resumption post Delta Learning
**Description:**
This scenario tests the full architectural loop from pause to resumption without a catastrophic restart. Following the successful cluster extraction and evaluation in UAT-03, the mocked DFT oracle returns ground truth forces. The system uses this to awaken the MACE model, generating a cloud of surrogate structures. The `PacemakerTrainer` performs an incremental delta update, mixing the new surrogate data with a replay buffer retrieved from the SQLite state manager. The newly compiled `.yace` potential is swapped into the LAMMPS simulation. The test verifies that LAMMPS resumes execution from the exact timestep where it was paused, retaining the previous velocity distributions and thermodynamic ensembles, rather than resetting to step zero or suffering a massive energy discontinuity (soft-started via Langevin thermostat).
The successful execution of this final test scenario proves the complete resolution of the "Time-Continuity Break" and "Catastrophic Forgetting" constraints. Upon receiving the single, high-fidelity ground truth calculation from the Quantum Espresso module, the orchestrator must immediately bypass the standard training loop and invoke the `FinetuneManager`. The test will assert that the final layers of the mocked MACE foundation model are briefly unfrozen and updated, allowing the model to quickly internalize the specific quantum mechanics of the newly discovered defect structure. Using this "awakened" model, the orchestrator generates a massive dataset of surrogate structural permutations by applying targeted `rattle` and strain perturbations to the failed configuration. Concurrently, the test must verify that the SQLite state database successfully executes a D-optimal stratified sampling query, returning a robust replay buffer comprised of historically validated bulk and interfacial structures. The `PacemakerTrainer` must then be shown to execute an explicit incremental update, loading the exact `.yace` binary generated during the Phase 1 zero-shot distillation and calculating delta-residuals against the combined surrogate and replay buffer datasets. Once the updated `.yace` potential is compiled, the test will trace the orchestrator's execution flow back into the LAMMPS `Engine Layer`. Here, we must definitively prove the Master-Slave inversion. The Python script will command the dummy LAMMPS subprocess to issue a `read_restart` directive, targeting the precise binary dump generated at the exact moment of the initial failure. The new potential coefficients will be dynamically injected via `pair_coeff` commands. Crucially, the test must verify that a highly damped `fix langevin` thermostat is temporarily activated for a predefined number of steps, gracefully absorbing any instantaneous kinetic energy spikes caused by the sudden shift in the underlying potential energy surface. Finally, the test asserts that the molecular dynamics iteration counter seamlessly increments from the previous pause state, flawlessly continuing the long-timescale trajectory without requiring a complete restart from $T=0$.

## 2. Behavior Definitions

**Feature:** Zero-Shot Baseline Distillation
**GIVEN** a valid `config.yaml` defining elements 'Fe' and 'O' with `DistillationConfig.enable=True`
**AND** the system is executing in a high-performance computing environment with access to massive combinatorial configuration spaces
**WHEN** the orchestrator is executed and phase one initializes
**THEN** the system generates an exhaustive combinatorial structure pool containing unary, binary, and explicitly defected ternary supercells
**AND** strictly filters it using the `MACEManager` foundation model based upon the quantified distance metrics within the latent feature space
**AND** guarantees that absolutely no calls are made to the expensive `DFTManager`
**AND** utilizes the highly confident subset to train a foundational `base.yace` polynomial potential augmented by a strict Lennard-Jones repulsive core
**AND** the potential ultimately passes the Validator subsystem checks for elastic tensors and positive-definite phonon dispersion frequencies on all requested parent bulk structures, proving the system can bootstrap itself entirely without prior quantum mechanical data.

**Feature:** Two-Tier Evaluator for Thermal Noise
**GIVEN** an active molecular dynamics simulation governed by the PyAceMaker Python framework utilizing the master-slave `fix python/invoke` inversion architecture
**AND** the configuration `ActiveLearningThresholds.smooth_steps` is strictly set to an integer value of 3
**WHEN** the simulated uncertainty spikes dramatically above the designated threshold for exactly 2 consecutive integration steps due to a violent but ultimately benign local thermal fluctuation
**THEN** the `TwoTierEvaluator` correctly logs a "Thermal Noise Detected, Ignoring Signal" diagnostic warning to the primary execution log
**AND** the internal `HaltTrigger` state machine flag remains strictly evaluated as `False`
**AND** the molecular dynamics simulation continues processing integration timesteps without ever pausing, completely bypassing the computationally ruinous density functional theory extraction and delta-learning pipelines, thereby guaranteeing long-timescale simulation throughput.

**Feature:** Intelligent Cutout and Auto-Passivation
**GIVEN** a permanently paused LAMMPS simulation containing a confirmed structural anomaly formally isolated at atom index 42 by the two-tier evaluation framework
**WHEN** the sophisticated cutout and passivation procedure is invoked by the master orchestrator thread
**THEN** the system mathematically extracts a spherical cluster centered precisely on Cartesian coordinates of atom 42
**AND** applies `force_weight=1.0` exclusively to the inner core atoms while assigning exactly `0.0` to the surrounding structural buffer
**AND** executes a topological graph traversal using the internal neighbor list to identify broken coordination environments
**AND** dynamically adds the necessary parameterized passivation atoms (e.g., fractional 'H') to satisfy the exposed dangling bonds
**AND** the `MACEManager` performs a localized LBFGS pre-relaxation, freezing the core coordinates while allowing the buffer to minimize its energy landscape
**AND** the resulting `Atoms` object is validated to be fully periodic, neutrally charged, and surrounded by a sufficient vacuum boundary before being passed to the Quantum Espresso driver.

**Feature:** Seamless MD Resume
**GIVEN** a newly trained, delta-updated `.yace` machine learning interatomic potential generated via the surrogate data explosion and SQLite replay buffer combination
**AND** a suspended LAMMPS simulation state immutably preserved via a high-fidelity binary `.restart` file generated at the exact moment of failure
**WHEN** the master orchestrator commands the simulation engine to resume its trajectory
**THEN** LAMMPS dynamically executes the `read_restart` directive and hot-reloads the new pair coefficients
**AND** continues time integration from the exact paused timestep $t_p$, preserving all massive spatial arrays, atomic velocities, and complex thermodynamic ensembles
**AND** strictly applies a temporary Langevin thermalization thermostat for the first $N$ steps to computationally damp any instantaneous energy divergence caused by the underlying polynomial shift, completely solving the "Time-Continuity Break" limitation.


## 3. Tutorial Strategy

To ensure a flawless user experience and facilitate rapid adoption, the scenarios defined above will be consolidated into a single, interactive, and reproducible tutorial format. The tutorial strategy relies on creating an executable Python file using `marimo`, a modern reactive notebook interface for Python.

The strategy incorporates two distinct execution modes:
1. **Mock Mode (Default):** The tutorial will use internal mock classes (e.g., `DummyMACEManager`, `DummyDFTManager`, `DummyLAMMPSEngine`) to simulate the heavy computational lifting. This allows a user to verify the orchestrator logic, state transitions, and file generation entirely on a standard laptop without requiring GPUs, MPI, or external API keys. It serves as a rapid, fail-safe verification of the system's structural integrity.
2. **Real Mode:** By toggling a configuration flag, users with full HPC environments can run the exact same tutorial invoking real LAMMPS C++ binaries and Quantum Espresso executables.

The tutorial will linearly guide the user through Phase 1 to Phase 4, explicitly highlighting the logs where thermal noise is ignored, where clusters are passivated, and where the MD seamlessly resumes.

## 4. Tutorial Plan

We will create a **SINGLE** executable file: `tutorials/UAT_AND_TUTORIAL.py`.

This file will be a Marimo notebook formatted as a standard Python script. It will encompass all UAT scenarios:
1. **Quick Start Setup:** Defining the PyAceConfig programmatically.
2. **Phase 1 Execution (UAT-01):** Demonstrating the zero-shot distillation and validating the output `base.yace`.
3. **Phase 2 & 3 Execution (UAT-02 & 03):** Running a miniature MD loop, showing the Two-Tier Evaluator ignoring a 2-step noise spike, then intentionally failing it with a massive structural distortion to trigger the Intelligent Cutout and Auto-Passivation. Visualizations of the extracted and passivated cluster will be generated using `ase.visualize`.
4. **Phase 4 Execution (UAT-04):** Demonstrating the Delta Learning process and the seamless resumption of the MD state object.

By consolidating this into one file, we prevent tutorial fragmentation and ensure the user sees the complete "Hierarchical Distillation" pipeline from start to finish.

## 5. Tutorial Validation

To validate the tutorial, the `UAT_AND_TUTORIAL.py` file must be executable sequentially without errors in the standard Python runtime (`uv run python tutorials/UAT_AND_TUTORIAL.py`). Due to headless environment constraints, standard CI will execute the script natively via Python rather than relying on `marimo run`. The validation is successful if the script runs to completion, all assertions pass, and the final "MD Simulation Successfully Resumed and Completed" message is printed to stdout, alongside the generation of dummy `.yace` and checkpoint artifacts in the specified temporary working directory.
