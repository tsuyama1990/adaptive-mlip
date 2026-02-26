# System Architecture: PyAceMaker

## 1. Summary

**PyAceMaker** is a cutting-edge, automated pipeline designed to revolutionize the generation of Machine Learning Interatomic Potentials (MLIPs). Its primary mission is to democratize high-accuracy atomistic simulations by streamlining the complex process of potential fitting. Traditionally, developing a robust interatomic potential—whether it be a classical force field or a modern MLIP like ACE (Atomic Cluster Expansion)—requires deep domain expertise, manual curation of datasets, and an iterative trial-and-error process that consumes significant human and computational resources. PyAceMaker automates this entire lifecycle, from initial structure generation to final model validation, ensuring reproducibility and efficiency.

The core philosophy of PyAceMaker is "Knowledge Distillation from Foundation Models." In the era of large-scale pre-trained models, starting from scratch is often inefficient. PyAceMaker leverages **MACE (Multi-ACE)**, a state-of-the-art equivariant graph neural network potential that has been pre-trained on vast datasets (like the Materials Project). Instead of performing thousands of expensive Density Functional Theory (DFT) calculations to explore the potential energy surface (PES), PyAceMaker uses a pre-trained MACE model as a "Surrogate Oracle." This surrogate guides the exploration of the configuration space, identifying high-uncertainty regions that truly require accurate DFT labeling.

The system implements a sophisticated **7-Step MACE Distillation Workflow**:
1.  **DIRECT Sampling**: Utilizes the DIRECT (DIviding RECTangles) algorithm to maximize entropy in the descriptor space, generating a diverse initial pool of atomic structures without any prior physics knowledge.
2.  **Uncertainty-based Active Learning**: Evaluating the "uncertainty" of the MACE model on the sampled structures. Only those configurations where the model is "confused" (high variance in ensemble predictions or internal metrics) are sent to the expensive DFT calculator. This drastically reduces the number of required DFT calculations, often by an order of magnitude.
3.  **MACE Fine-tuning**: The selected high-value structures are used to fine-tune the MACE model itself, adapting it from a general-purpose model to a specialist for the specific chemical system under study.
4.  **Surrogate Data Generation**: The fine-tuned MACE model, now a reliable approximation of the local PES, runs molecular dynamics (MD) simulations to generate thousands of "surrogate" structures. This allows for massive sampling of thermal fluctuations and phase space at near-zero computational cost compared to ab initio MD.
5.  **Surrogate Labeling**: These thousands of structures are labeled by the MACE model (energy, forces, virial stress), creating a large, synthetic dataset.
6.  **Pacemaker Base Training**: A lightweight, fast ACE potential is trained on this large synthetic dataset. This "Base Model" learns the general topology of the PES captured by MACE.
7.  **Delta Learning (The Crown Jewel)**: Finally, the system performs "Delta Learning." The Base ACE model is fine-tuned using the small but high-precision set of DFT data collected in Step 2. The model learns to correct the systematic errors (the "delta") between the MACE approximation and the ground-truth DFT.

The result is a **PyAceMaker** potential that combines the best of all worlds: the general chemical intuition of a foundation model, the extreme accuracy of specific DFT calculations, and the millisecond-scale inference speed of the ACE formalism. This architecture not only saves computational costs but also provides a systematic, bias-free method for potential construction, suitable for users ranging from novice students to expert computational material scientists.

## 2. System Design Objectives

The design of PyAceMaker is governed by a set of strict objectives to ensure it meets the demands of modern scientific research and software engineering standards.

### 2.1. Maximum Automation & Minimum Intervention
The primary objective is to reduce human intervention to zero after the initial configuration. The user should define the *intent* (e.g., "I want a potential for the SN2 reaction of CH3Cl") in a simple YAML configuration file, and the system should handle the rest. This implies robust error handling, self-healing workflows (e.g., restarting failed calculations), and sensible default parameters that work for 90% of use cases. The "Zero-Config" ideal means the system must infer reasonable bounds for hyperparameters like cutoff radii and sampling temperatures based on the elemental composition.

### 2.2. Cost Efficiency via Active Learning
Computational resources (CPU/GPU hours) are a finite budget. A key success metric for PyAceMaker is the "Accuracy per DFT Calculation" ratio. The system is designed to treat DFT calls as the most expensive resource. The Active Learning module must be ruthless in rejecting redundant structures. If a structure is already well-described by the current model, calculating its DFT energy is a waste. The architecture supports "Budget Constraints," allowing users to set a hard limit on the number of DFT evaluations (e.g., "Max 500 DFT calls"), forcing the system to prioritize the most informative data points.

### 2.3. Modularity and Interchangeability
The field of MLIPs is moving fast. Today MACE is state-of-the-art; tomorrow it might be Allegro or NequIP. Similarly, ACE might be superseded by a newer linear basis. Therefore, the architecture must be modular. The `Oracle` (source of truth), `Generator` (sampling engine), and `Trainer` (model fitting) are defined as abstract interfaces.
-   **Oracle Agnosticism**: The system can switch between VASP, Quantum Espresso, CASTEP, or even a Lennard-Jones toy model (for testing) without changing the core orchestration logic.
-   **Surrogate Agnosticism**: While "MACE Distillation" is the default, the design allows swapping MACE for another graph network or Gaussian Process model.
-   **Output Agnosticism**: The pipeline currently targets ACE potentials, but the labeled datasets produced are format-agnostic (XYZ, ASE Atoms) and can be used to train any other MLIP.

### 2.4. Robustness and Reproducibility
Scientific software must be reproducible. PyAceMaker enforces strict versioning of datasets and models. Every artifact generated (initial structures, DFT inputs, trained models) is saved with a cryptographic hash of the configuration that produced it. The `Orchestrator` ensures idempotency: if the pipeline crashes at Step 4, restarting it should resume exactly from Step 4, verifying the integrity of Steps 1-3 rather than re-running them. This is crucial for long-running workflows on High-Performance Computing (HPC) clusters where wall-time limits are common.

### 2.5. Scalability
The system must handle systems ranging from a simple diatomic molecule to complex interfaces with hundreds of atoms. This requires efficient memory management (streaming large datasets instead of loading them entirely into RAM) and parallel execution support. The `DFTManager` component is designed to interface with workload managers like Slurm or PBS, submitting array jobs for parallel DFT evaluation, rather than running them sequentially on the head node.

## 3. System Architecture

The high-level architecture of PyAceMaker is a "Hub-and-Spoke" model centered around the `Orchestrator`. The Orchestrator manages the state of the workflow and delegates tasks to specialized sub-systems.

### 3.1. Core Components

1.  **Orchestrator (`pyacemaker.orchestrator`)**:
    The brain of the operation. It reads the `config.yaml`, initializes the necessary components, and executes the 7-step lifecycle. It manages the `GlobalState`, tracking which steps are complete and where the artifacts are stored.

2.  **Domain Models (`pyacemaker.domain_models`)**:
    Strict Pydantic definitions of data structures. This includes `StructureContainer` (wrapping ASE Atoms with metadata), `TrainingConfig`, `ActiveLearningConfig`, and `GenerationRequest`. These models ensure type safety and validation at the boundaries of every module.

3.  **Structure Generator (`pyacemaker.structure_generator`)**:
    Responsible for exploring the configuration space.
    -   `DirectSampler`: Implements the DIRECT algorithm for initial entropy maximization.
    -   `MDSampler`: Runs Molecular Dynamics (using ASE) to sample thermal distributions.
    -   `MutationEngine`: Applies random rattlings and cell deformations (legacy support).

4.  **Oracle System (`pyacemaker.oracle`)**:
    The source of truth and uncertainty.
    -   `MaceSurrogate`: Wraps the MACE model to provide energies, forces, and *uncertainties*. It acts as a filter.
    -   `DFTManager`: Interfaces with ab initio codes (VASP/QE) to generate ground-truth labels. It handles input file generation, execution, and output parsing.

5.  **Trainer Engine (`pyacemaker.trainer`)**:
    Manages the learning process.
    -   `MaceFinetuner`: Adapts the pre-trained MACE model to the active set.
    -   `PacemakerWrapper`: Wraps the `pacemaker` library to train ACE potentials.
    -   `DeltaTrainer`: Implements the specialized loss function for Delta Learning, weighting DFT data significantly higher than surrogate data.

### 3.2. Data Flow

The data flow is cyclic and iterative. Structures flow from Generator -> Oracle (Filter) -> DFT (Label) -> Trainer -> Generator (Better Sampling).

1.  **Config & Init**: User provides composition and constraints.
2.  **Exploration**: Generator produces candidate structures.
3.  **Query**: Oracle evaluates candidates.
    -   If Uncertainty > Threshold -> Send to DFT Queue.
    -   If Uncertainty < Threshold -> Use Surrogate Label (or discard).
4.  **Labeling**: DFT Queue is processed, returning `(E, F, V)` (Energy, Forces, Virial).
5.  **Learning**:
    -   MACE is updated with new DFT data.
    -   ACE is trained/updated with the composite dataset (Surrogate + DFT).
6.  **Validation**: Model performance is checked against a hold-out test set.
7.  **Loop**: If target accuracy is not met, return to Step 2 with the improved models.

### 3.3. Architecture Diagram

```mermaid
graph TD
    User[User / Config] -->|1. Initialize| Orch[Orchestrator]

    subgraph "Core Logic"
        Orch -->|2. Request Sampling| Gen[Structure Generator]
        Orch -->|3. Query Uncertainty| Oracle[Oracle Interface]
        Orch -->|7. Train Models| Trainer[Trainer Engine]
    end

    subgraph "External/Surrogate"
        Gen -->|DIRECT Algo| Pool[Candidate Pool]
        Oracle -->|MACE Prediction| MACE[MACE Surrogate]
        MACE -->|High Uncertainty| ActiveSet[Active Set]
        MACE -->|Low Uncertainty| SurrogateData[Surrogate Dataset]
    end

    subgraph "Ground Truth"
        ActiveSet -->|Submit Job| DFT[DFT Calculator (VASP/QE)]
        DFT -->|Truth Labels| RefData[Reference Dataset]
    end

    subgraph "Model Building"
        RefData -->|Fine-tune| MACE
        SurrogateData -->|Base Training| ACE[Pacemaker ACE]
        RefData -->|Delta Learning| ACE
    end

    MACE -.->|Guide MD| Gen
```

## 4. Design Architecture

The codebase is structured to promote separation of concerns and maintainability. We use a standardized directory structure and strict typing.

### 4.1. Directory Structure (ASCII Tree)

```text
pyacemaker/
├── pyproject.toml              # Dependencies and Tool Config
├── README.md                   # Entry point documentation
├── config.yaml                 # Default/Example configuration
├── src/
│   └── pyacemaker/
│       ├── __init__.py
│       ├── main.py             # CLI Entry point
│       ├── orchestrator.py     # Main workflow controller
│       ├── constants.py        # Global constants (units, defaults)
│       ├── core/               # Abstract Base Classes & Interfaces
│       │   ├── __init__.py
│       │   ├── interfaces.py   # Protocol definitions (Oracle, Generator, Trainer)
│       │   └── exceptions.py   # Custom exception hierarchy
│       ├── domain_models/      # Pydantic Data Models
│       │   ├── __init__.py
│       │   ├── config.py       # Configuration schemas
│       │   ├── data.py         # Structure and Dataset models
│       │   └── state.py        # Workflow state tracking
│       ├── structure_generator/# Sampling Logic
│       │   ├── __init__.py
│       │   ├── direct.py       # DIRECT sampling implementation
│       │   ├── md.py           # Molecular Dynamics sampler
│       │   └── mutation.py     # Random mutation logic
│       ├── oracle/             # Energy/Force Evaluators
│       │   ├── __init__.py
│       │   ├── mace_wrapper.py # MACE model wrapper
│       │   ├── dft_manager.py  # DFT code abstraction
│       │   └── uncertainty.py  # Active Learning query strategies
│       ├── trainer/            # Model Training Logic
│       │   ├── __init__.py
│       │   ├── ace_trainer.py  # Pacemaker wrapper
│       │   ├── mace_trainer.py # MACE finetuning logic
│       │   └── delta.py        # Delta learning specific logic
│       └── utils/              # Shared Utilities
│           ├── __init__.py
│           ├── io.py           # File I/O (xyz, json, yaml)
│           ├── logging.py      # Structured logging
│           └── parallel.py     # Multiprocessing helpers
├── tests/                      # Pytest Suite
│   ├── unit/
│   ├── integration/
│   └── conftest.py
└── dev_documents/              # Documentation & Specs
```

### 4.2. Key Data Models

The system relies heavily on **Pydantic** for data validation.

*   `GlobalConfig`: The root configuration object, parsed from YAML. It contains sub-configs for `DistillationConfig`, `DFTConfig`, `ACEConfig`, etc. It validates that paths exist, numerical values are within physical bounds (e.g., temperature > 0), and required keys are present.
*   `AtomStructure`: A wrapper around `ase.Atoms` that adds provenance metadata. It tracks *how* the structure was generated (DIRECT, MD, Mutation), its `uncertainty_score`, and its status (Candidate, Labeled, Failed).
*   `Dataset`: Represents a collection of `AtomStructure` objects. It supports lazy loading to handle large datasets that don't fit in memory.
*   `WorkflowState`: A JSON-serializable object that tracks the progress of the pipeline. It stores which steps are completed, paths to current best models, and the iteration count. This enables the "resume" functionality.

### 4.3. Interface Design

We use Python `Protocol` or `ABC` to define clear contracts.

*   `IOracle`:
    ```python
    class IOracle(ABC):
        @abstractmethod
        def compute(self, structure: AtomStructure) -> LabeledStructure: ...
        @abstractmethod
        def compute_batch(self, structures: List[AtomStructure]) -> List[LabeledStructure]: ...
    ```
*   `IGenerator`:
    ```python
    class IGenerator(ABC):
        @abstractmethod
        def generate(self, count: int, context: GenerationContext) -> List[AtomStructure]: ...
    ```
*   `ITrainer`:
    ```python
    class ITrainer(ABC):
        @abstractmethod
        def train(self, dataset: Dataset, previous_model: Optional[Path] = None) -> Path: ...
    ```

## 5. Implementation Plan

The development is divided into 6 strictly sequential cycles.

### Cycle 01: Core Framework & DIRECT Sampling (Step 1)
**Goal**: Establish the project skeleton and implement the first step of the workflow.
-   **Features**:
    -   Setup `src` structure, `pyproject.toml`, and basic logging.
    -   Implement `domain_models` for Config and Structures.
    -   Implement `Orchestrator` basic loop (stateless).
    -   Implement `structure_generator.direct` to perform DIRECT sampling on simple descriptors.
    -   Create a "Mock Oracle" (Lennard-Jones) to test the flow without heavy deps.
-   **Deliverable**: A script that reads a config and outputs a diverse set of XYZ structures using DIRECT sampling.

### Cycle 02: MACE Oracle & Active Learning (Step 2)
**Goal**: Integrate the "Brain" of the operation.
-   **Features**:
    -   Implement `oracle.mace_wrapper` to load MACE models.
    -   Implement `oracle.uncertainty` to calculate uncertainty metrics (variance, committee).
    -   Implement the "Active Learning Filter" logic in the Orchestrator.
    -   Connect `oracle.dft_manager` (interface only, with mock implementation) to "label" high-uncertainty structures.
-   **Deliverable**: A pipeline that takes Cycle 01 structures, filters them by MACE uncertainty, and "labels" the top N% using the mock oracle.

### Cycle 03: MACE Surrogate Loop (Steps 3 & 4)
**Goal**: Close the loop with MACE fine-tuning and usage.
-   **Features**:
    -   Implement `trainer.mace_trainer` to fine-tune MACE on the "labeled" data from Cycle 02.
    -   Implement `structure_generator.md` to run MD simulations using the *fine-tuned* MACE model.
    -   This enables the generation of physically relevant structures (Step 4) that are impossible to find via random sampling.
-   **Deliverable**: A loop that fine-tunes MACE and then uses it to generate thousands of MD snapshots.

### Cycle 04: Surrogate Labeling & Base ACE Training (Steps 5 & 6)
**Goal**: Create the dataset and train the target ACE model.
-   **Features**:
    -   Implement batch labeling in `oracle.mace_wrapper` to label the thousands of MD structures (Step 5).
    -   Implement `trainer.ace_trainer` to wrap `pacemaker`.
    -   Convert MACE-labeled data into Pacemaker-compatible formats (`.pdata` / `.xyz`).
    -   Execute the Base Training (Step 6).
-   **Deliverable**: A trained `.yace` potential file derived purely from MACE distillation.

### Cycle 05: Delta Learning (Step 7)
**Goal**: Achieve DFT accuracy via Delta Learning.
-   **Features**:
    -   Implement `trainer.delta` logic.
    -   Modify `ace_trainer` to accept a reference potential (the Base Model) and a sparse high-accuracy dataset (DFT).
    -   Implement the loss function weighting strategy (High weight for DFT, Low/Zero for regularization).
    -   This step corrects the systematic errors of the MACE surrogate.
-   **Deliverable**: A "Delta-Corrected" `.yace` potential that matches DFT accuracy on the active set.

### Cycle 06: Full Orchestration & SN2 Polish
**Goal**: Production-ready system and User Acceptance Testing.
-   **Features**:
    -   Wire all 7 steps together in the `Orchestrator` with full error handling and state persistence (resume capability).
    -   Implement the SN2 Reaction specific configuration and tutorial.
    -   Refine CLI arguments and logging for end-users.
    -   Final performance optimization (caching, parallel I/O).
-   **Deliverable**: The final `pyacemaker` package and the `UAT_AND_TUTORIAL.py` demonstrating the SN2 case study.

## 6. Test Strategy

Testing is continuous and multi-layered.

### Cycle 01 Test Strategy
-   **Unit Tests**: Verify `DirectSampler` generates structures within valid bounds. Check Pydantic models for correct validation of invalid inputs.
-   **Integration**: Run the Orchestrator with `step1_direct_sampling` only. Verify that an output XYZ file is created and contains the requested number of atoms.

### Cycle 02 Test Strategy
-   **Unit Tests**: Mock the MACE model to return predictable uncertainty values. Verify the sorting and filtering logic of the Active Learning module.
-   **Integration**: Feed a known set of structures to the Oracle. Assert that only the "high uncertainty" ones are flagged for calculation. Verify the interface with the Mock DFT calculator.

### Cycle 03 Test Strategy
-   **Unit Tests**: Verify the MACE fine-tuning wrapper correctly calls the underlying training commands/API.
-   **Integration**: Perform a full "Surrogate Loop". Train MACE on a small dataset -> Run MD -> Verify the MD trajectory is physically reasonable (e.g., no exploding atoms). This confirms the fine-tuned model is stable.

### Cycle 04 Test Strategy
-   **Unit Tests**: Verify the data conversion (ASE Atoms -> Pacemaker format). Check that the `PacemakerWrapper` constructs valid command-line arguments.
-   **Integration**: Train a Base ACE model on a small synthetic dataset. Run a simple evaluation (e.g., energy prediction) to ensure the model is essentially functional (not checking accuracy yet, just mechanics).

### Cycle 05 Test Strategy
-   **Unit Tests**: Verify the Delta Learning configuration. Ensure weights are applied correctly in the training config generation.
-   **Integration**: "The Accuracy Test". Train a Base Model, then apply Delta Learning with a shifted dataset. Verify that the final model predicts values closer to the shifted dataset than the Base Model. This proves the "Delta" correction is working.

### Cycle 06 Test Strategy
-   **System Testing (UAT)**: Execute the full SN2 Reaction scenario in "Mock Mode". Verify that all 7 steps complete without error, intermediate files are cleaned up (or saved as requested), and the final `.yace` file is valid.
-   **Performance Testing**: Profile the `Orchestrator` to ensure memory usage is stable during the large-scale Surrogate Labeling phase (Step 5).
-   **Documentation Verification**: Ensure the `UAT_AND_TUTORIAL.py` runs exactly as described in the README.
