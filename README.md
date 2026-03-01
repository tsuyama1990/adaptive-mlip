# PyAceMaker NextGen

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-DRAFT-orange.svg)
![Version](https://img.shields.io/badge/version-2.1.0-brightgreen.svg)

**Adaptive machine learning interatomic potentials construction orchestrator, featuring Hierarchical Distillation and Master-Slave Inversion.**

## Overview

**PyAceMaker** is an automated workflow tool designed to construct robust Machine Learning Interatomic Potentials (MLIPs) for practical High-Performance Computing (HPC) environments.

Manually constructing MLIPs for long-timescale phenomena is tedious and error-prone. Version 2.1.0 completely revamps the Active Learning cycle. By introducing "Hierarchical Distillation" from foundation models (MACE) and "Master-Slave Resume" capabilities for Molecular Dynamics, PyAceMaker guarantees continuous MD simulations, prevents catastrophic forgetting during retraining, and intelligently repairs extracted clusters before passing them to Quantum Espresso.

## Key Features

*   **Master-Slave MD Resume**: MD simulations no longer reset to time zero when encountering unknown structures. The system pauses LAMMPS, updates the potential, and seamlessly resumes from the exact microsecond and coordinate state where it left off.
*   **Intelligent Cluster Extraction & Passivation**: Extracts spherical regions around uncertain "epicentre" atoms, pre-relaxes the boundary using MACE while freezing the core, and auto-passivates dangling bonds to prevent DFT divergence.
*   **Two-Tier Uncertainty Evaluation**: Separates the threshold to pause MD (`threshold_call_dft`) from the threshold to select training data (`threshold_add_train`). This provides strong resistance against false halts triggered by harmless thermal noise.
*   **Hierarchical Distillation & Delta Learning**: Replaces expensive $O(N)$ batch retraining. Fine-tunes the MACE foundation model using sparse DFT data, explosively generates thousands of surrogate data points, and incrementally updates the ACE potential using a Replay Buffer to prevent catastrophic forgetting.

## Architecture Overview

PyAceMaker operates as a 4-Phase state machine orchestrated around an event-driven core. The architecture strictly enforces dependency injection and streaming data structures to handle millions of atoms without memory exhaustion.

```mermaid
flowchart TD
    A[Initial State] --> Phase1

    subgraph Phase1 [Phase 1: Zero-Shot Distillation]
    B1[Generate Combinatorial Structures] --> B2[DIRECT Sampling]
    B2 --> B3[MACE Confidence Filtering]
    B3 --> B4[Pacemaker Baseline Train (LJ Delta)]
    end

    Phase1 --> Phase2

    subgraph Phase2 [Phase 2: Validation]
    C1[EOS & Phonon Calc] --> C2{Stable?}
    C2 -- No --> B1
    C2 -- Yes --> C3[Miniature MD Stress Test]
    end

    Phase2 --> Phase3

    subgraph Phase3 [Phase 3: Intelligent Cutout]
    D1[LAMMPS MD Simulation] --> D2{Max Gamma > threshold_call_dft?}
    D2 -- No --> D1
    D2 -- Yes --> D3[Identify Epicentre Atoms > threshold_add_train]
    D3 --> D4[Spherical Cutout & Weighting]
    D4 --> D5[MACE Pre-Relaxation Buffer]
    D5 --> D6[Auto-Passivation]
    D6 --> D7[Clean DFT Calc (QE)]
    end

    Phase3 --> Phase4

    subgraph Phase4 [Phase 4: Hierarchical Fine-Tuning]
    E1[Finetune MACE with DFT Data] --> E2[Generate Surrogate Data via Awakened MACE]
    E2 --> E3[Incremental ACE Train + Replay Buffer]
    E3 --> E4[Master-Slave Resume LAMMPS MD]
    end

    E4 --> D1
```

## Prerequisites

*   **Python**: >= 3.11
*   **Package Manager**: `uv`
*   **DFT Code**: Quantum Espresso (`pw.x` executable in PATH)
*   **MLIP Trainer**: Pacemaker (`pace_train`, `pace_activeset` executables in PATH)
*   **MD Engine**: LAMMPS Python Interface (`lammps` package, with `USER-PACE` support)
*   **Foundation Model**: PyTorch and MACE (`mace-mp-0`)

## Installation & Setup

We strictly use `uv` for lightning-fast dependency management and environment isolation.

```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
uv sync
```

## Usage

1.  **Prepare Configuration**:
    Create a `config.yaml` file defining your project parameters, including the new Distillation and Workflow settings.

    ```yaml
    project_name: "FePt_Alloy"
    structure:
        elements: ["Fe", "Pt"]
        supercell_size: [2, 2, 2]
    distillation:
        enable: true
        mace_model_path: "mace-mp-0-medium"
        uncertainty_threshold: 0.05
    workflow:
        incremental_update: true
        replay_buffer_size: 500
        thresholds:
            threshold_call_dft: 0.05
            threshold_add_train: 0.02
            smooth_steps: 3
    cutout:
        core_radius: 4.0
        buffer_radius: 3.0
        enable_pre_relaxation: true
        enable_passivation: true
    dft:
        code: "quantum_espresso"
        # ...
    ```

2.  **Run PyAceMaker**:

    ```bash
    # Dry run to validate config
    uv run pyacemaker --config config.yaml --dry-run

    # Start the 4-Phase active learning loop
    uv run pyacemaker --config config.yaml
    ```

## Development Workflow

This project enforces strict code quality standards using `ruff` and `mypy`.
We build iteratively through defined Development Cycles (Cycle 01 - 04).

*   **Run Linters**:
    ```bash
    uv run ruff check
    ```
*   **Run Type Checker**:
    ```bash
    uv run mypy .
    ```
*   **Run Tests**:
    ```bash
    uv run pytest
    ```

## Project Structure

```
src/pyacemaker/
├── core/               # Core logic (Master-Slave Engine, TieredOracle, FinetuneManager)
├── domain_models/      # Strict Pydantic schemas (Config, Distillation, Workflow)
├── interfaces/         # External code drivers (Quantum Espresso, Pacemaker)
├── utils/              # Helper functions (Intelligent Extraction, Passivation)
├── factory.py          # Dependency Injection Container
├── orchestrator.py     # Event-driven 4-Phase state machine
└── main.py             # CLI entry point
```

## License

MIT License
