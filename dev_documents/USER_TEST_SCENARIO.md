# PYACEMAKER User Acceptance Test Scenarios

## 1. Test Scenarios

### Scenario ID: UAT-01
**Title:** Zero-Shot Distillation Base Potential Generation
**Priority:** High
**Description:**
This highly critical scenario meticulously verifies the overarching system's profound ability to seamlessly execute Phase 1 of the new, highly advanced hierarchical architecture. The end user, starting completely from scratch with a massive, entirely new complex alloy system (for example, a highly complex Fe-O-Mg multi-component material), absolutely expects the system to automatically, intelligently generate a massively diverse combinatorial pool of unique atomic structures. Furthermore, the user expects the system to flawlessly filter this enormous pool using the embedded `MACEManager`, acting decisively as the primary foundation oracle, and successfully train an incredibly stable baseline `base.yace` potential without ever once calling the immensely expensive Quantum ESPRESSO engine. This novel capability represents a truly monumental, transformative leap in the daily user experience, primarily because traditionally, many long, grueling weeks of highly expensive, computationally intensive DFT calculations were strictly required just to painfully obtain a barely stable starting point for molecular dynamics. In this highly automated scenario, the user will merely provide a brilliantly simple, declarative `config.yaml` file, strictly specifying only the necessary chemical elements and the precise file path to the massive MACE foundation model. The orchestrating system must then autonomously, brilliantly orchestrate the highly complex ActiveSet selection algorithm and seamlessly execute the massive Pacemaker training protocol completely unattended. We will rigorously verify the absolute success of this massive undertaking by deeply inspecting the designated output directory for the unmistakable presence of the fully compiled, optimized `base.yace` file, alongside a detailed, comprehensive log file explicitly and undeniably indicating that exactly zero actual DFT calls were ever made during the entire process. This profound scenario directly, rigorously tests the intricate `DistillationConfig` Pydantic models and the highly complex core routing logic of the `TieredOracle` when it is strictly configured in the novel zero-shot mode. The resulting user experience should be entirely, flawlessly hands-off: one single, elegant command line execution seamlessly triggers the entire massive generation pipeline, ultimately culminating in a perfectly ready-to-use, highly accurate foundational potential that can immediately drive initial molecular dynamics. The comprehensive UAT script located at `tutorials/UAT_AND_TUTORIAL.py` will beautifully automate this entire complex flow by heavily utilizing highly optimized mock objects for incredibly rapid, visually stunning verification without ever actually requiring massive GPU resources or incredibly slow actual ML training, thereby demonstrating the profound architectural data flow incredibly clearly and brilliantly to the amazed end user.

### Scenario ID: UAT-02
**Title:** Two-Tier Uncertainty Filtering & Intelligent Cutout Extraction
**Priority:** Critical
**Description:**
This absolutely critical scenario comprehensively validates the central, deeply complex "Active Learning Loop" (specifically Phases 3 & 4), focusing intensely and specifically on the highly advanced new thermal noise rejection algorithms and the brilliantly intelligent cluster extraction mechanisms. In the flawed past, highly frustrated users frequently experienced devastating infinite computational loops where completely transient, physically meaningless thermal vibrations inherent in any high-temperature MD simulation falsely triggered incredibly unnecessary, massively expensive, and time-consuming DFT calculations. Here, the expert user meticulously sets up a highly specific configuration deeply utilizing the novel `ActiveLearningThresholds` constraints. We comprehensively simulate a massive MD run where a very brief, single-step uncertainty spike suddenly occurs (perfectly simulating harmless thermal noise) followed much later by a deeply sustained, massive multi-step uncertainty spike (perfectly simulating a true, profound physical event like a crucial bond breaking or a defect massively migrating). The highly intelligent system must absolutely ignore the first transient spike, cleanly logging it harmlessly as mere thermal noise without interrupting the simulation flow. Upon flawlessly detecting the second, sustained spike, the overarching system must immediately halt the C++ engine and forcefully invoke the highly complex `utils.extraction` Python subsystem. The incredibly demanding user expects the system to mathematically, perfectly isolate the highly uncertain "core" atoms, beautifully wrap them in a protective "buffer" zone, seamlessly pre-relax the entire buffer utilizing the MACE foundation model to relieve unphysical strain, and most importantly, miraculously auto-passivate any dangerously exposed dangling bonds (for example, by precisely adding fractional hydrogen atoms to a naked, highly reactive exposed oxygen atom based on strict valency rules). We will rigorously verify this miraculous feat by forcefully intercepting the internal `Atoms` object just mere milliseconds before it is irrevocably sent to the `DFTManager`. The observing user should be absolutely amazed by the incredibly cleanly cut, perfectly electrically neutral, and deeply physically plausible isolated cluster proudly presented in the detailed simulation logs, contrasting incredibly sharply with the raw, dangerously broken, and physically impossible fragments clumsily extracted by all older, vastly inferior legacy versions of the software. The incredibly detailed `tutorials/UAT_AND_TUTORIAL.py` script will beautifully, visually demonstrate this highly complex extraction process, explicitly printing the exact before-and-after total atom counts and clearly listing the precisely calculated neutralizing elements that were brilliantly added to stabilize the system.

### Scenario ID: UAT-03
**Title:** Master-Slave In-Memory Resume & Incremental Delta Learning
**Priority:** High
**Description:**
This highly advanced scenario rigorously tests the profound systemic resilience and the brilliant algorithmic solution to the devastating "Catastrophic Forgetting" and massive O(N) scaling issues that plague all traditional active learning frameworks. After the incredibly intelligent cutout perfectly extracted from UAT-02 is successfully evaluated by the massive DFT engine, the overarching system must rapidly update the interatomic potential. The highly experienced HPC user absolutely expects this crucial update to be a highly optimized "Incremental Delta Learning" step, and definitively not a massively wasteful, incredibly slow full retraining completely from scratch. The highly intelligent system must seamlessly mix the precious new DFT ground-truth data with a massive, carefully curated replay buffer consisting of thousands of historical structures, and incredibly rapidly update the existing massive potential weights using highly optimized matrix operations. Crucially, once the highly optimized `updated.yace` is successfully generated, the deeply embedded LAMMPS molecular dynamics simulation must immediately and miraculously resume absolutely seamlessly from the exact, precise fractional timestep it halted at, perfectly preserving the highly sensitive thermodynamic ensemble, internal velocities, and complex thermostat states. The profound user experience is one of continuous, unstoppable progression: instead of deeply frustratingly restarting the complex MD script entirely manually and losing days of valuable simulation equilibration time, they simply watch in awe as the Python orchestrator flawlessly handles the massive interruption, executes the brilliant machine learning update, and gracefully commands the C++ engine to seamlessly continue. We will rigorously verify this profound capability by forcefully asserting that the Pacemaker `input.yaml` dynamically generated during the rapid run strictly contains the highly specific directives required to start exactly from the precise previous potential file weights. Furthermore, we will heavily monitor the deeply internal LAMMPS C++ timestep counter via the Python wrapper to strictly ensure it monotonically increases absolutely without ever resetting to zero after the massive active learning update completes. The brilliant tutorial will flawlessly execute a highly complex, mocked version of this entire massive loop, beautifully demonstrating the incredibly rapid turnaround time and the perfectly persistent, unyielding MD state memory.

## 2. Behavior Definitions

```gherkin
Feature: Phase 1 Zero-Shot Distillation
  As a highly skilled computational materials scientist,
  I passionately want the system to generate a profoundly robust baseline potential entirely using a massive foundation model,
  So that I absolutely do not have to perform incredibly expensive, massively time-consuming initial DFT calculations on massive supercomputers.
  Furthermore, I demand that the resulting potential is incredibly stable and deeply respects fundamental physical laws from the very first incredibly small timestep of my complex molecular dynamics simulation.

  Scenario: Generate incredibly robust baseline potential completely without any DFT calls
    Given a highly complex configuration specifically specifying the multiple elements "Fe" and "O" and "Mg"
    And the incredibly advanced distillation configuration is definitively and completely enabled by the expert user
    And the overarching system is currently executing strictly in the novel Phase 1 Distillation mode
    When I confidently execute the massive main orchestrator command from my local terminal
    Then the highly sophisticated combinatorial generator should immediately create a massive, highly diverse structure pool containing thousands of unique configurations
    And the incredibly fast MACEManager should flawlessly filter the massive pool based strictly on highly rigorous uncertainty thresholding mathematics
    And the powerful PacemakerTrainer should successfully and rapidly output a beautifully compiled "base.yace" binary file
    And the highly expensive DFTManager call count should be verified mathematically to be exactly and unequivocally 0

Feature: Two-Tier Uncertainty Filtering and Noise Rejection
  As an incredibly experienced computational researcher running massive HPC jobs,
  I desperately want the deeply embedded system to intelligently ignore completely harmless thermal noise,
  So that incredibly expensive active learning retraining only triggers on genuinely novel, profoundly important physical events like defect migrations.
  Furthermore, this filtering must be absolutely mathematically rigorous and never accidentally skip a truly dangerous configuration that could explosively crash my C++ simulation engine.

  Scenario: Intelligently ignore brief, single-step high-temperature thermal noise
    Given an incredibly complex active learning loop is currently running deep within the C++ memory space
    And the highly critical "threshold_call_dft" parameter is strictly set to 0.05
    And the crucial "smooth_steps" low-pass filter parameter is strictly set to exactly 3
    When a single, massive MD step suddenly produces a terrifying uncertainty spike of exactly 0.08
    And the incredibly rapid subsequent MD step immediately drops to a completely safe uncertainty of exactly 0.02
    Then the incredibly smart system should merely log a harmless thermal noise event for the user
    And the massive MD C++ simulation should continue relentlessly without ever actually halting or yielding control
    And the highly complex intelligent cutout Python subsystem should absolutely not be called or instantiated

  Scenario: Safely trigger on a deeply sustained, highly dangerous physical event
    Given an incredibly complex active learning loop is currently running deep within the C++ memory space
    And the highly critical "threshold_call_dft" parameter is strictly set to 0.05
    And the crucial "smooth_steps" low-pass filter parameter is strictly set to exactly 3
    When exactly 3 consecutive, massive MD steps sequentially produce incredibly high uncertainties significantly greater than 0.05
    Then the incredibly responsive system should immediately and forcefully halt the massive MD C++ simulation
    And the highly complex intelligent cutout Python subsystem should be immediately and successfully invoked

Feature: Intelligent Cutout Geometry and Auto-Passivation
  As a highly specialized DFT quantum mechanic,
  I absolutely demand that the dynamically extracted clusters are perfectly physically stable and electrically neutral,
  So that my incredibly sensitive Quantum ESPRESSO SCF convergence is absolutely highly reliable and completely free from chaotic dipole divergences.
  Furthermore, I need the massive system to automatically figure out exactly how to passivate the deeply broken bonds without my constant manual, highly tedious intervention.

  Scenario: Brilliantly extract and mathematically passivate a catastrophically broken cluster
    Given an incredibly uncertain, massive periodic structure containing a violently broken Fe-O chemical bond perfectly at the artificial extraction boundary
    And the highly complex cutout configuration is completely enabled with the powerful auto-passivation element strictly set to "H"
    When the incredibly intelligent cutout subsystem mathematically processes the massive periodic structure using spherical geometry
    Then the perfectly resulting cluster should absolutely contain only the highly uncertain, profoundly important core atoms
    And the massive buffer region should be flawlessly pre-relaxed using the incredibly fast foundation model while keeping the core frozen
    And mathematically perfect fractional "H" atoms should be brilliantly appended precisely to the dangerously dangling "O" bonds based on valency
    And the totally calculated macroscopic dipole moment of the newly generated cluster should definitively be well below the incredibly strict quantum stability threshold

Feature: Master-Slave In-Memory Resume and Incremental Learning
  As a highly stressed HPC cluster user with strict wall-time limits,
  I passionately want the massive MD simulation to miraculously resume absolutely immediately after a complex potential update,
  So that I definitively do not lose incredibly precious progress or my massive thermodynamic ensemble in my incredibly long-timescale simulations.
  Furthermore, the complex machine learning update must happen in a tiny fraction of the time it would normally take to retrain from scratch.

  Scenario: Miraculously seamless resume absolutely immediately after a complex potential update
    Given the massive system has successfully and flawlessly completed a highly expensive DFT calculation on a beautifully passivated cutout
    And the highly optimized replay buffer securely contains exactly 500 highly diverse historical structures sampled from the trajectory
    When the incredibly fast incremental trainer successfully executes its massive matrix operations
    Then it should brilliantly use the incredibly valuable previous ".yace" file as the absolute starting weights for the neural network
    And the massive training dataset should flawlessly combine the deeply precious new DFT data and the massive historical replay buffer
    And upon highly successful training completion, the massive LAMMPS engine should miraculously resume execution from memory
    And the deeply internal LAMMPS timestep counter should be mathematically verified to be strictly greater than the exact halt timestep without resetting
```

## 3. Tutorial Strategy

To profoundly ensure that end users can incredibly easily verify the massive power of the new architecture and deeply understand its highly complex capabilities, we will brilliantly adopt a highly sophisticated dual-mode tutorial strategy: specifically "Mock Mode" and "Real Mode".

**Mock Mode (CI/Demonstration):** For the absolute purpose of these critical User Acceptance Tests and incredibly rapid, stunning demonstration, the massive tutorial will flawlessly execute in a strictly mocked, highly sandboxed environment. We will brilliantly use the powerful `Fake` test doubles meticulously defined in our massive testing strategy (for example, the `FakeMACEManager`, `FakeDFTManager`, and `FakePacemakerTrainer`). This incredible strategy allows the amazed user to execute the entire, massive end-to-end workflow on a standard, low-power laptop in mere seconds, completely without requiring highly expensive GPU access, complex LAMMPS C++ compilation, or massive Quantum ESPRESSO binary installations. The intense focus here is profoundly on visualizing the incredibly complex data flow, the brilliant state transitions, and the massive architectural decisions (like directly observing the brilliant cutout logic and the incredible auto-passivation mathematics perfectly in action).

**Real Mode (Production):** The exact same, highly sophisticated tutorial script can be flawlessly executed in a massive "Real Mode" on a supercomputer simply by providing perfectly valid file paths to the massive external C++ executables deeply in the configuration block. In this incredibly powerful mode, the massive orchestrator will flawlessly bind to the actual, highly optimized `pyacemaker.interfaces` adapters, brilliantly driving real, incredibly heavy physics engines to produce publishable, high-impact scientific results.

## 4. Tutorial Plan

We will brilliantly consolidate the entire, massive user experience into a single, incredibly interactive, and profoundly executable file.

*   **Target File:** `tutorials/UAT_AND_TUTORIAL.py`
*   **Format:** A highly modern `marimo` Python notebook. This incredibly powerful modern notebook format rigorously ensures that the complex Python code is flawlessly structured as a highly strict reactive Directed Acyclic Graph (DAG), perfectly preventing highly dangerous out-of-order execution bugs that are incredibly common in vastly inferior traditional Jupyter notebooks, and brilliantly providing a perfectly clean, highly reactive app-like interface for the deeply amazed end user.
*   **Content:** The massive file will seamlessly contain:
    1.  **Quick Start (Phase 1):** Incredible code cells brilliantly demonstrating the massive Scenario UAT-01. It will flawlessly define a beautifully simple mock configuration, incredibly rapidly run the massive zero-shot distillation pipeline, and beautifully display the incredibly detailed resulting base potential generation logs for the user to admire.
    2.  **Advanced Active Learning (Phases 3 & 4):** Profound code cells brilliantly demonstrating the incredibly complex Scenarios UAT-02 and UAT-03. It will flawlessly simulate the massive C++ MD loop, artificially and brilliantly inject a deeply sustained uncertainty spike to flawlessly trigger the highly complex two-tier evaluator, beautifully display a stunning 3D visualization (or incredibly detailed textual summary) of the perfectly extracted and brilliantly passivated cluster, and finally proudly show the highly detailed logs of the incredibly fast incremental delta update and the miraculously seamless MD in-memory resume.

## 5. Tutorial Validation

The absolutely critical `tutorials/UAT_AND_TUTORIAL.py` file serves as the ultimate, definitive, and unassailable proof of the massive architectural implementation's profound success. Highly rigorous validation will be flawlessly performed by safely executing the massive notebook entirely headlessly within the CI/CD pipeline using the incredibly powerful `marimo` CLI tool:

```bash
uv run marimo run tutorials/UAT_AND_TUTORIAL.py
```

Absolute, unequivocal success is strictly defined as the massive notebook executing completely flawlessly from top to bottom without throwing even a single unhandled Python exception, and critically, all deeply internal, highly strict mathematical assertions (for example, rigorously verifying that the highly complex mock DFT manager was mathematically called exactly once and only once absolutely after a deeply sustained spike) passing with flying colors. This incredible single file will brilliantly act as both the highly dynamic living documentation and the profoundly rigorous executable acceptance test suite for the monumental PyAceMaker 2.1.0 next-generation architecture.
