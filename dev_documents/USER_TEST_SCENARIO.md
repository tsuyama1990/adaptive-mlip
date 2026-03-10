# PyAceMaker User Acceptance Test Scenarios

## 1. Test Scenarios

The following scenarios are designed to validate the core features of the PyAceMaker platform from a user's perspective. The goal is to ensure a smooth, verifiable, and powerful user experience, emphasizing the "Hierarchical Distillation" architecture and robust MD simulations.

### Scenario 01: Zero-Shot Baseline Distillation [Priority: High]
Objective: The user defines a simple chemical space (e.g., Fe-O). The system automatically generates a diverse structure pool, filters it using the MACE foundation model (without calling DFT), and trains a foundational baseline Atomic Cluster Expansion (ACE) potential. The user experiences an incredibly fast cold start to the active learning loop, moving from an empty directory to a functioning baseline potential in seconds instead of hours. The tutorial will demonstrate this by configuring a pure distillation step and examining the output potential file and log messages indicating MACE confidence filtering. This scenario is incredibly important because it showcases the immediate value proposition of the PyAceMaker platform. Users are accustomed to spending weeks or even months generating initial DFT datasets just to bootstrap a basic machine learning potential. By leveraging the zero-shot distillation capability of the MACE foundation model, the user witnesses the platform instantly bypassing this massive bottleneck. The test will guide the user through setting up a minimal configuration file, specifying the target elements, and launching the orchestrator in distillation mode. The user will then verify that the platform successfully generates a diverse set of crystalline, defective, and strained structures completely autonomously. They will inspect the logs to confirm that the MACEManager evaluates these structures, correctly rejecting those with high uncertainty and keeping only the most reliable data points. Finally, the user will verify the creation of the compiled ACE potential file (`base.yace`) in the output directory, proving that the system has successfully distilled the foundation model's broad knowledge into a fast, linear-scaling potential ready for immediate molecular dynamics simulation, all without consuming a single cycle of expensive quantum mechanical compute time.

### Scenario 02: Thermal Noise Rejection via Two-Tier Thresholds [Priority: High]
Objective: The user initiates an active learning MD loop on a relatively stable lattice at elevated temperatures. The simulation encounters typical thermal vibrations that cause minor, momentary spikes in uncertainty. The system evaluates these against the threshold_call_dft and the smooth_steps requirement. The user observes in the logs that these momentary spikes are correctly identified as thermal noise and ignored, preventing the MD loop from continuously halting and calculating unnecessary DFT data. The MD continues uninterrupted, validating the noise-rejection paradigm. This scenario demonstrates the sophisticated signal processing capabilities embedded within the active learning loop. In traditional systems, high-temperature simulations are notoriously difficult to manage because natural atomic vibrations constantly trigger uncertainty thresholds, leading to endless, pointless retraining cycles that waste computational resources and stall the simulation progress. To test this, the user will configure a molecular dynamics simulation with a high target temperature and deliberately set a relatively sensitive primary uncertainty threshold. The simulation will be launched, and the user will monitor the real-time telemetry output. They will observe instances where the instantaneous uncertainty briefly spikes above the threshold due to extreme thermal fluctuations. Crucially, the user will verify through the logs that the TieredOracle's smoothing logic correctly intercepts these spikes, preventing the generation of a halt signal because the high uncertainty was not sustained across the required number of consecutive steps. This proves to the user that the PyAceMaker platform possesses the intelligence to distinguish between harmless, transient noise and genuine physical novelty, ensuring that the active learning loop only pauses for truly important, structurally significant events, thereby maximizing computational efficiency and simulation throughput.

### Scenario 03: Intelligent Cutout and Auto-Passivation [Priority: Critical]
Objective: The user injects a severe defect (e.g., an unphysical vacancy or colliding atoms) into a large MD supercell. The simulation halts due to genuine, sustained uncertainty. The system isolates the exact epicenter atom, extracting it along with a defined buffer radius. Crucially, the user observes that the extracted cluster is not just a raw cut, but a meticulously prepared chunk where the boundaries are relaxed using MACE, and broken bonds are automatically passivated with fractional dummy atoms (like Hydrogen). The user can inspect the generated .xyz file of the extracted cluster to verify its electrical neutrality and structural safety before the simulated DFT calculation takes place. This scenario validates the most critical and complex algorithmic feature of the platform. Handling massive multi-million atom simulations requires extracting localized regions of interest, but naive extraction invariably creates unphysical dangling bonds that cause DFT calculations to crash or yield garbage data. The user will manually construct a highly defective configuration, such as a severe grain boundary collision, and force the active learning loop to process it. Upon halting, the user will intercept the workflow and deeply inspect the output of the extraction utility. They will load the generated `.xyz` cluster file into a molecular visualization tool (like OVITO or VESTA) to visually confirm that the core region remains perfectly preserved, while the surrounding buffer region has been smoothly relaxed. Most importantly, the user will visually verify the presence of the automatically placed passivation atoms (e.g., Hydrogen caps) precisely at the boundary locations where chemical bonds were severed during extraction. This provides absolute proof that the platform is generating physically sound, electrically neutral, and safely calculable isolated clusters, guaranteeing the stability and accuracy of the subsequent quantum mechanical evaluations.

### Scenario 04: Master-Slave Seamless MD Resume [Priority: High]
Objective: The user runs a long MD simulation that successfully triggers a halt due to a new structural configuration. After the Orchestrator performs the background DFT calculation, surrogate data generation, and incremental update of the ACE potential, it commands LAMMPS to resume. The user verifies in the logs and output trajectories that the MD engine did not restart from step zero. Instead, it picked up exactly at the halted step number, preserving the system's kinetic energy, velocities, and time continuity. This demonstrates true Master-Slave inversion. The ability to seamlessly resume a molecular dynamics simulation is the holy grail of long-timescale active learning. Without it, simulations are constantly reset, preventing the observation of slow, complex physical processes. In this scenario, the user will configure a standard active learning loop and allow it to naturally encounter a novel structural state that triggers a legitimate halt. The user will note the exact timestep at which the halt occurred (e.g., step 45,200). After the orchestrator automatically completes the hierarchical finetuning and potential update processes, the simulation will automatically restart. The user will then rigorously inspect the LAMMPS log files and the generated trajectory `.dump` files. They will definitively verify that the simulation timestamps continue exactly from 45,201 onwards, rather than resetting to 0. Furthermore, by inspecting the thermodynamic output, the user will confirm that the total energy, temperature, and velocity distributions of the system remain perfectly continuous and consistent across the halt-resume boundary. This absolute proof of time continuity and kinetic preservation validates the Master-Slave inversion architecture, assuring the user that PyAceMaker is capable of driving true, unbroken, long-timescale virtual experiments.

## 2. Behavior Definitions

The following Gherkin-style definitions formalize the expected behaviors for automated and manual testing verification. These definitions serve as the absolute source of truth for the system's intended functionality, providing clear, unambiguous, and testable criteria for every major architectural feature. By framing these requirements in the widely understood Given/When/Then syntax, we bridge the gap between technical implementation and user expectations, ensuring that the software perfectly aligns with the desired scientific workflows. These behavior definitions are not merely documentation; they form the very foundation of our automated End-to-End (E2E) test suite, guaranteeing that future code modifications never break these critical operational contracts. The Zero-Shot Distillation feature ensures that the system can bootstrap itself entirely from a foundation model without requiring any initial quantum mechanical data. The Two-Tier Thermal Noise Rejection feature guarantees that the active learning loop remains highly efficient, refusing to waste expensive computational cycles on trivial high-temperature atomic vibrations. The Intelligent Cutout and Passivation feature dictates the precise mechanical steps required to safely extract a local region from a massive supercell, ensuring that the resulting cluster is physically sound, structurally relaxed at the boundaries, and chemically neutralized to prevent DFT convergence failures. Finally, the Master-Slave Seamless Resume feature enforces the strict requirement for absolute time continuity, mandating that molecular dynamics simulations must always continue exactly from where they paused, preserving all kinetic history and momentum, thereby allowing scientists to observe true long-timescale physical phenomena without arbitrary interruptions or resets.

**Feature: Zero-Shot Distillation**
```gherkin
GIVEN a clean PyAceMaker workspace
AND a configuration specifying the Fe-O chemical system with distillation_mode: true
WHEN the orchestrator is initialized for iteration 0
THEN it should generate a combinatorial structure pool
AND it should query the MACEManager Oracle for confidence scores
AND it should train a baseline potential using only structures where uncertainty is below the configured threshold
AND it should never invoke the DFTManager
```

**Feature: Two-Tier Thermal Noise Rejection**
```gherkin
GIVEN a running PyAceMaker MD loop using a trained potential
AND a configuration with threshold_call_dft set to 0.05 and smooth_steps set to 3
WHEN the MD engine reports a momentary uncertainty spike of 0.06 for exactly 1 step
THEN the TieredOracle should classify it as thermal noise
AND the MD simulation should NOT halt
AND the system should not trigger a DFT calculation
```

**Feature: Intelligent Cutout and Passivation**
```gherkin
GIVEN a large MD supercell that has halted due to high uncertainty
AND the system has identified an epicenter atom index
WHEN the extract_intelligent_cluster utility is invoked
THEN it should extract a cluster based on the configured core_radius and buffer_radius
AND it should assign force_weight=1.0 to the core atoms and 0.0 to the buffer atoms
AND it should apply a pre-relaxation to the buffer atoms using the specified Mock or MACE calculator while keeping the core fixed via ase.constraints.FixAtoms
AND it should automatically append passivation elements (e.g., 'H') to any undercoordinated surface atoms
```

**Feature: Master-Slave Seamless Resume**
```gherkin
GIVEN a halted MD simulation that stopped at step 15000 due to uncertainty
AND the Orchestrator has successfully completed the Incremental Update phase
WHEN the Orchestrator commands the LammpsEngine to continue the simulation
THEN the LammpsEngine should launch a script containing a resume_from_step: 15000 directive
AND it should inject a 'fix langevin' command for the initial soft-start steps
AND the resulting LAMMPS log should show the simulation continuing from step 15000 rather than step 0
AND the velocity distribution should remain consistent with the pre-halt state
```

## 3. Refinement: Master Plan for User Acceptance Testing and Tutorials

To provide users with an engaging, interactive, and reproducible way to verify these scenarios, the entire UAT suite will be consolidated into a single executable tutorial.

### 3.1 Tutorial Strategy
The tutorial strategy employs a Mock Mode approach. Since running actual MACE neural networks and Quantum Espresso DFT calculations requires significant hardware (GPUs, HPC clusters) and hours of compute time, the tutorial will leverage the robust mock implementations (like `FakeLammpsDriver` and `FakePacemakerBinary`) already planned in the `pyacemaker.core` tests. This allows any user to execute the full orchestration pipeline—from zero-shot distillation to intelligent cutout and seamless resume—on a standard laptop in seconds, completely independent of external binaries.

### 3.2 Tutorial Plan
We will create a **SINGLE** Marimo Python notebook file located at `tutorials/UAT_AND_TUTORIAL.py`.

This file will be structured as a step-by-step interactive guide:
1. Environment Setup: Defines the mock configurations and sets up temporary directories (`tempfile.TemporaryDirectory`) to prevent filesystem pollution.
2. Scenario 01 (Zero-Shot): Triggers the Orchestrator's cold start and prints the logs showing MACE filtering.
3. Scenario 02 (Noise Rejection): Simulates an MD result with a short uncertainty spike and verifies the Orchestrator ignores it using the `TieredOracle`.
4. Scenario 03 (Cutout & Passivation): Manually invokes `extract_intelligent_cluster` on a dummy defective lattice and renders the resulting .xyz data to visualize the core, buffer, and newly added passivation atoms.
5. Scenario 04 (Seamless Resume): Runs a mocked loop iteration that forces a halt, performs an update, and prints the generated LAMMPS script to prove the `resume_from_step` and `fix langevin` logic is correctly injected.

### 3.3 Tutorial Validation
The file `tutorials/UAT_AND_TUTORIAL.py` must be written as a valid Python script compatible with marimo. It will be strictly typed and linted according to the pyproject.toml standards (excluding specific name rules where explicitly ignored). Validation of the tutorial consists of executing it natively (`uv run python tutorials/UAT_AND_TUTORIAL.py`) to ensure all assertions pass and logs are generated correctly, providing a flawless out-of-the-box user experience.