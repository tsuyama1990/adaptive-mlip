# PYACEMAKER System Architecture Document

## 1. Summary

The **PYACEMAKER** (High-efficiency MLIP Construction & Operation System) project represents a significant leap forward in the field of computational materials science, specifically addressing the challenges associated with constructing and operating Machine Learning Interatomic Potentials (MLIPs). Traditionally, the creation of high-fidelity MLIPs has been the domain of experts possessing deep knowledge in both data science and quantum physics. The process involved manual, iterative cycles of structure generation, Density Functional Theory (DFT) calculations, potential fitting, and validation—a workflow prone to human error, inefficiency, and "extrapolation risk" where potentials fail catastrophically in unknown regions of the chemical space.

PYACEMAKER aims to democratise this technology by automating the entire lifecycle of an ACE (Atomic Cluster Expansion) potential. At its core, the system utilizes the "Pacemaker" engine but wraps it in a robust, autonomous orchestration layer. This Orchestrator manages the complex interplay between exploration (finding new atomic configurations), labelling (calculating their true energy using DFT), learning (fitting the potential), and inference (running simulations). By integrating these distinct phases into a seamless, self-correcting loop, the system achieves a "Zero-Config Workflow" where a user need only provide a high-level configuration file (e.g., "I want a potential for Fe-Pt alloys") and the system handles the rest.

A key innovation of PYACEMAKER is its emphasis on "Physics-Informed Robustness". Unlike purely data-driven approaches that may behave unpredictably outside their training data, this system enforces physical laws through Delta Learning. It uses a robust baseline potential (such as Lennard-Jones or ZBL) to handle short-range core repulsion, ensuring that atoms never fuse together even in high-energy collision scenarios—a common failure mode in standard MLIPs. Furthermore, the system employs active learning strategies, specifically D-Optimality (Active Set optimization), to select only the most information-rich structures for DFT calculation. This drastically reduces the computational cost, achieving high accuracy with a fraction of the data required by random sampling methods.

The system architecture is designed to be modular and scalable, supporting a range of simulation backends from classical Molecular Dynamics (MD) with LAMMPS to long-timescale Adaptive Kinetic Monte Carlo (aKMC) with EON. This allows the generated potentials to be used not just for short-term thermal properties, but for studying slow phenomena like diffusion, phase transitions, and surface catalysis over macroscopic timescales. By bridging the gap between accuracy (DFT) and speed (MD/kMC), PYACEMAKER serves as a powerful accelerator for materials discovery.

## 2. System Design Objectives

The design of the PYACEMAKER system is guided by several critical objectives, constraints, and success criteria, ensuring it meets the needs of both novice users and power users in research environments.

### 2.1 Goals

1.  **Zero-Config Automation**: The primary goal is to eliminate the need for users to write Python scripts or manage complex file dependencies manually. A single YAML configuration file should be sufficient to launch a complete active learning campaign. The system must intelligently infer reasonable defaults for hyperparameters (like temperature schedules or mixing betas) based on the material's properties.

2.  **Data Efficiency**: DFT calculations are computationally expensive. The system aims to maximise the information gain per calculation. By using uncertainty quantification (the extrapolation grade, $\gamma$), the system identifies "knowledge gaps" and only triggers expensive DFT calculations for structures that significantly improve the potential. The target is to achieve production-quality accuracy (RMSE Energy < 1 meV/atom) with 90% fewer calculations than random sampling.

3.  **Physical Reliability**: A major constraint in MLIPs is physical validity. The system must guarantee that the potential is stable. This includes ensuring no imaginary phonon modes in equilibrium structures, satisfying Born stability criteria for elastic constants, and preventing "holes" in the potential energy surface where atoms could collapse. The implementation of hybrid potentials (ACE + Baseline) is non-negotiable to ensure safety in high-temperature simulations.

4.  **Self-Healing Resilience**: In automated workflows, individual calculation failures (e.g., SCF convergence errors in Quantum Espresso) are inevitable. The system must possess "autonomic" capabilities to detect these failures and attempt self-repair strategies (e.g., adjusting smearing, mixing parameters) without halting the entire pipeline.

### 2.2 Constraints

*   **Computational Resources**: The system must run effectively on a range of hardware, from a single workstation (for development/testing) to HPC clusters (for production). Docker/Singularity containerisation is required to manage the complex dependency tree (LAMMPS, QE, Python libraries).
*   **Latency**: The "On-the-Fly" (OTF) learning loop requires low latency. The time between detecting a high-uncertainty structure and updating the potential should be minimised. This constrains the choice of learning algorithms (using Fine-tuning instead of training from scratch) and DFT settings (using small, embedded clusters).
*   **Compatibility**: The system must output potentials in standard formats compatible with LAMMPS, the de facto standard for MD simulations. It must also interface seamlessly with ASE (Atomic Simulation Environment) for general structure manipulation.

### 2.3 Success Criteria

*   **Automation Level**: A user can start a run and return days later to a converged potential without manual intervention.
*   **Robustness**: The system survives 1,000,000 MD steps on a complex alloy system without segmentation faults or non-physical explosions.
*   **Accuracy**: The final potential reproduces DFT phonon bands and elastic constants within 5-10% error.
*   **Usability**: A new user can install and run the "Hello World" tutorial (Fe/Pt on MgO) in under 30 minutes on a standard laptop (in Mock mode).

## 3. System Architecture

The PYACEMAKER architecture follows a modular, cyclic design pattern centered around a central **Orchestrator**. This Orchestrator manages the flow of data between four specialized autonomous agents: the **Structure Generator** (Explorer), the **Oracle** (Labeller), the **Trainer** (Learner), and the **Dynamics Engine** (Consumer).

### 3.1 High-Level Diagram

```mermaid
graph TD
    subgraph "Orchestration Layer"
        ORC[Orchestrator]
        CFG[Config Manager]
        LOG[State Logger]
    end

    subgraph "The Explorer"
        SG[Structure Generator]
        M3G[M3GNet/Universal Potential]
        POL[Exploration Policy]
    end

    subgraph "The Oracle"
        DFT[DFT Manager]
        QE[Quantum Espresso]
        EMB[Periodic Embedding]
    end

    subgraph "The Trainer"
        PM[Pacemaker Wrapper]
        AS[Active Set Selector]
        DL[Delta Learning (LJ/ZBL)]
    end

    subgraph "The Engine"
        MD[MD Interface (LAMMPS)]
        KMC[kMC Interface (EON)]
        UQ[Uncertainty Watchdog]
    end

    subgraph "The Guardian"
        VAL[Validator]
        PH[Phonon/Elasticity Check]
    end

    User[User] -->|config.yaml| CFG
    CFG --> ORC
    ORC -->|1. Request Candidates| SG
    SG -->|Candidate Structures| ORC
    ORC -->|2. Select Best| AS
    AS -->|Active Set| ORC
    ORC -->|3. Compute Truth| DFT
    DFT -->|Forces & Energies| ORC
    ORC -->|4. Update Pot| PM
    PM -->|potential.yace| ORC
    ORC -->|5. Deploy & Run| MD
    MD -->|Stream Trajectory| ORC
    MD -- Uncertainty Limit --> UQ
    UQ -->|Halt & Extract| ORC
    ORC -->|6. Verify| VAL
    VAL -->|Report| User

    style ORC fill:#f9f,stroke:#333,stroke-width:2px
    style MD fill:#bbf,stroke:#333,stroke-width:2px
    style DFT fill:#bfb,stroke:#333,stroke-width:2px
```

### 3.2 Component Data Flow

1.  **Initialization**: The Orchestrator loads the `config.yaml`. If no initial potential exists, it calls the **Structure Generator** to create random or heuristic structures (using M3GNet for initial guesses) and sends them to the **Oracle**.
2.  **Training**: The **Oracle** performs DFT calculations (with self-healing and periodic embedding). The resulting labelled data is fed to the **Trainer**, which fits the ACE potential (using Delta Learning logic) and selects the most important data points via Active Set optimization.
3.  **Exploration & Inference**: The trained potential is deployed to the **Dynamics Engine**. LAMMPS runs the simulation. The **Uncertainty Watchdog** monitors the extrapolation grade ($\gamma$).
4.  **Interruption & Learning**: If $\gamma$ exceeds a threshold, the simulation halts. The high-uncertainty local environment is extracted, embedded into a new simulation cell, and sent back to the **Oracle** for labelling. The potential is updated (fine-tuned), and the simulation resumes.
5.  **Validation**: Periodically, or upon convergence, the **Validator** runs physical checks (Phonons, Elasticity) to ensure the potential is not just numerically accurate but physically meaningful.

## 4. Design Architecture

The system is implemented in Python 3.11+, utilising strong typing (Pydantic) to enforce data validity across module boundaries. The file structure separates concerns clearly, with a dedicated `src` directory for source code and `dev_documents` for specifications.

### 4.1 File Structure

```ascii
pyacemaker/
├── pyproject.toml              # Dependencies and Linter Config
├── README.md                   # Project Landing Page
├── src/
│   └── pyacemaker/
│       ├── __init__.py
│       ├── main.py             # Entry point
│       ├── config.py           # Pydantic Configuration Models
│       ├── orchestrator.py     # Central Control Logic
│       ├── core/
│       │   ├── __init__.py
│       │   ├── generator.py    # Structure Generation Logic
│       │   ├── oracle.py       # DFT/ASE Interface
│       │   ├── trainer.py      # Pacemaker Wrapper
│       │   ├── engine.py       # MD/LAMMPS Interface
│       │   └── validator.py    # Quality Assurance
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── embedding.py    # Periodic Embedding Algo
│       │   ├── delta.py        # LJ/ZBL Parameter Gen
│       │   └── io.py           # File handling helpers
│       └── interfaces/
│           ├── __init__.py
│           ├── lammps_driver.py
│           ├── qe_driver.py
│           └── eon_driver.py
└── tests/
    ├── __init__.py
    ├── test_orchestrator.py
    ├── test_generator.py
    ├── test_oracle.py
    ├── test_trainer.py
    ├── test_engine.py
    └── conftest.py
```

### 4.2 Class Design Overview

*   **`PyAceConfig` (Pydantic Model)**: The single source of truth for configuration. It validates the user's YAML input, checking ranges (e.g., temperature > 0) and paths.
*   **`Orchestrator`**: The state machine. It maintains the current cycle number, iteration count, and the status of the active learning loop. It uses the Strategy Pattern to delegate tasks to the core modules.
*   **`DFTManager`**: Encapsulates the complexity of Quantum Espresso. It implements the "Self-Healing" pattern, wrapping the calculation in a retry loop that modifies input parameters upon failure.
*   **`MDInterface`**: Abstraction over the LAMMPS Python bindings. It manages the construction of the `in.lammps` script dynamically, injecting the hybrid pair styles and `fix halt` commands.
*   **`PeriodicEmbedder`**: A utility class responsible for the geometric transformation of cutting a cluster from a bulk/surface and placing it into a periodic box suitable for DFT.

## 5. Implementation Plan

The development is divided into 8 sequential cycles. Each cycle builds upon the previous one, ensuring a stable and testable increment of functionality.

### Cycle 01: Foundation & Configuration
**Objective**: Establish the project skeleton, configuration management, and abstract interfaces.
**Features**:
*   Setup of `pyproject.toml`, directory structure, and CI/CD basics.
*   Implementation of `PyAceConfig` using Pydantic to parse and validate `config.yaml`.
*   Implementation of the `Orchestrator` class with a dummy loop.
*   Definition of Abstract Base Classes (ABCs) for `Generator`, `Oracle`, `Trainer`, `Engine`.
*   Setup of logging infrastructure to track the system state.
*   **Verification**: The system can read a config file and print the planned execution flow without running real physics.

### Cycle 02: The Oracle (DFT Automation)
**Objective**: Implement the ability to run reliable DFT calculations.
**Features**:
*   Implementation of `DFTManager` using ASE's `Espresso` calculator.
*   Development of the input file generator (selecting pseudopotentials, k-points).
*   Implementation of the "Self-Healing" logic: catching `JobFailedException` and retrying with fallback parameters (e.g., higher smearing).
*   Implementation of the `PeriodicEmbedder` to transform arbitrary atom clusters into DFT-ready supercells.
*   **Verification**: The system can take a list of atomic structures and return their DFT energies and forces, recovering from induced errors.

### Cycle 03: The Explorer (Structure Generation)
**Objective**: Create the engine for exploring chemical space.
**Features**:
*   Implementation of `StructureGenerator`.
*   Integration with `M3GNet` (or a mock for testing) to provide initial stable structures ("Cold Start").
*   Implementation of perturbation policies: Random displacement (Rattling), Scaling (Strain), and Defect introduction (Vacancies).
*   Logic to output these candidates as ASE Atoms objects.
*   **Verification**: The system can generate a diverse set of valid atomic structures (perturbed bulk, surfaces) from a simple composition string (e.g., "FePt").

### Cycle 04: The Trainer (Pacemaker Integration)
**Objective**: Bridge the system with the Pacemaker library for potential fitting.
**Features**:
*   Implementation of `PacemakerWrapper`.
*   Logic to generate `config.yaml` specific to Pacemaker (cutoffs, basis set sizes).
*   Implementation of `DeltaLearning` utility: automatically calculating LJ/ZBL parameters for the baseline potential.
*   Integration of `pace_activeset` to filter training data based on D-optimality.
*   Execution of `pace_train` via subprocess or API.
*   **Verification**: The system can take a dataset of labelled structures and output a valid `potential.yace` file that combines ACE and the physical baseline.

### Cycle 05: The Engine (MD & Inference)
**Objective**: Enable molecular dynamics simulations with on-the-fly monitoring.
**Features**:
*   Implementation of `MDInterface` using `lammps` python module.
*   Dynamic generation of LAMMPS input scripts that use `pair_style hybrid/overlay`.
*   Implementation of the `UncertaintyWatchdog`: configuring `compute pace` and `fix halt` to stop simulation when $\gamma$ is high.
*   Parsing of LAMMPS log files to extract the exact frame and atom indices where failure occurred.
*   **Verification**: The system can run an MD simulation that automatically stops when it encounters a configuration outside the training data's validity domain.

### Cycle 06: The Orchestrator (Active Learning Loop)
**Objective**: Close the loop. Connect all components into an autonomous cycle.
**Features**:
*   Integration of Generator, Oracle, Trainer, and Engine into the `Orchestrator`'s main loop.
*   Implementation of the "Halt & Resume" logic: Extract halted structure -> Embed -> Label -> Train -> Resume.
*   Management of file versioning (iter_001, iter_002, ...).
*   State persistence: saving the loop state so it can be restarted after a crash.
*   **Verification**: The system runs a full active learning campaign on a simple system (e.g., Al) without human intervention, improving the potential over time.

### Cycle 07: The Guardian (Validation & QA)
**Objective**: Ensure the generated potentials are physically sound, not just numerically fit.
**Features**:
*   Implementation of `Validator` module.
*   Integration with `Phonopy` to calculate phonon band structures and check for imaginary modes (dynamical stability).
*   Implementation of Elastic Constant calculation ($C_{11}, C_{12}, C_{44}$) and Born stability checks.
*   EOS (Equation of State) curve generation and Bulk Modulus comparison.
*   Generation of an HTML `validation_report`.
*   **Verification**: The system produces a report flagging a potential as "UNSAFE" if it predicts unstable phonons, even if the energy error is low.

### Cycle 08: The Expander (kMC & Production)
**Objective**: Extend capabilities to long timescales and finalize the system.
**Features**:
*   Implementation of `EONWrapper` for Adaptive Kinetic Monte Carlo.
*   Integration of the "Fe/Pt on MgO" specific scenario requirements (complex multi-species interface).
*   Final polish of CLI and logging.
*   Comprehensive documentation generation.
*   **Verification**: The system successfully runs the full "Grand Challenge" scenario, simulating deposition and ordering of FePt on MgO.

## 6. Test Strategy

Testing is paramount for a system that claims "Zero-Config" reliability. We employ a pyramid testing strategy, utilizing `pytest` for unit and integration tests.

### Cycle 01 Testing
*   **Unit Tests**: Verify `PyAceConfig` correctly raises validation errors for invalid YAML inputs (e.g., negative temperatures). Verify the Singleton pattern of the `Logger`.
*   **Integration Tests**: Test the `Orchestrator`'s ability to load a mock configuration and instantiate all dummy modules without error.
*   **Strategy**: Use Dependency Injection to pass mock configurations.

### Cycle 02 Testing
*   **Unit Tests**: Test `PeriodicEmbedder` on simple geometries (e.g., single atom, dimer) to ensure correct cell padding.
*   **Integration Tests**: Run `DFTManager` against a "Mock Espresso" (a script that returns fixed random numbers) to test the file I/O pipeline.
*   **Hardware Tests**: Run a real, lightweight QE calculation (e.g., single H atom) in the CI environment to verify the binary interfaces.

### Cycle 03 Testing
*   **Unit Tests**: Verify that `StructureGenerator` creates structures with the correct chemical composition. Check that strain application results in the correct lattice parameter changes.
*   **Visual Tests**: Use `ase.visualize` (headless) to snapshot generated structures and manual verification.

### Cycle 04 Testing
*   **Unit Tests**: Verify `DeltaLearning` calculates correct LJ parameters for known elements.
*   **Integration Tests**: Mock the `pace_train` command to verify that the `PacemakerWrapper` constructs the correct command-line arguments and handles output paths correctly.

### Cycle 05 Testing
*   **Unit Tests**: Parse generated LAMMPS input scripts to ensure `pair_style hybrid/overlay` is correctly syntactically formed.
*   **Integration Tests**: Run a short LAMMPS simulation (using the library interface) with a dummy potential to verify the `fix halt` triggers correctly when a variable exceeds a threshold (simulate this by manually forcing the variable high).

### Cycle 06 Testing
*   **System Tests**: The "Loop Test". Run the orchestrator with Mocks for Oracle and Trainer. The Oracle returns a specific energy, the Trainer returns a "better" potential. Verify that the Orchestrator performs exactly N loops and stops.
*   **State Tests**: Interrupt the loop (Simulate Ctrl+C) and verify that it can resume from the last checkpoint `iter_XX`.

### Cycle 07 Testing
*   **Unit Tests**: Validation logic checks. Feed known bad data (e.g., imaginary frequencies) to the `Validator` and ensure it returns `False` / `Fail`.
*   **Integration Tests**: Run Phonopy on a perfect crystal (using a simple LJ potential) to verify the pipeline produces a valid band structure plot.

### Cycle 08 Testing
*   **User Acceptance Tests (UAT)**: The "Grand Challenge".
*   **Mock Mode**: Run the full Fe/Pt on MgO scenario with mocked DFT/Training (~5 mins). Verify all files are created.
*   **Real Mode**: (Manual) Run on a workstation. Verify physics (L10 ordering observed).
*   **Regression Tests**: Ensure that adding kMC functionality didn't break the pure MD workflow.
