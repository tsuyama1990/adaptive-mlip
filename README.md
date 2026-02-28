# PyAceMaker

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Verified-brightgreen.svg)

**NextGen Hierarchical Distillation Architecture for Adaptive Machine Learning Interatomic Potentials.**

## Overview

**PyAceMaker** is an advanced, automated workflow orchestration tool designed to construct highly robust Machine Learning Interatomic Potentials (MLIPs). Version 2.1.0 introduces a paradigm-shifting **Hierarchical Distillation Architecture** inspired by FLARE. It seamlessly bridges the gap between ultrafast Foundational Models (like MACE) and highly accurate ab-initio calculations (Quantum Espresso), enabling flawless, long-timescale Molecular Dynamics (MD) simulations scaling to millions of atoms in HPC environments.

### The Problem It Solves
Traditional active learning loops suffer from "MD Continuity Breaks" (restarting from step zero after retraining) and physical deterioration when extracting clusters from bulk systems (dangling bonds). PyAceMaker v2.1.0 solves this via **Master-Slave Inversion** (LAMMPS drives Python, ensuring seamless resume) and **Intelligent Cutout & Auto-Passivation** (chemically repairing extracted clusters before passing them to DFT).

## Key Features

*   **Seamless MD Continuity (Master-Slave Inversion):** MD simulations pause, trigger potential updates, and resume from the exact coordinate and velocity state. No lost time evolution.
*   **Intelligent Cutout & Passivation:** Automatically extracts high-uncertainty regions (the "epicenter"), performs MACE-driven boundary relaxation, and auto-passivates dangling bonds (e.g., adding Hydrogen) to ensure perfect DFT convergence.
*   **Hierarchical Distillation (Zero-Shot to DFT):** Employs MACE for rapid, zero-shot baseline generation and fast surrogate evaluation, restricting expensive Quantum Espresso calls only to critical, high-uncertainty structures.
*   **Incremental Delta Learning:** Prevents catastrophic forgetting and calculation explosions by using Replay Buffers and fitting only the *difference* between consecutive potentials in $O(1)$ time.
*   **Thermal Noise Exclusion:** A two-tier threshold system filters out transient thermal vibrations, ensuring the MD halts only for genuine model uncertainty.
*   **HPC Resilience & Artifact Cleanup:** Granular, task-level JSON/SQLite checkpointing survives node crashes, while an automated daemon aggressively cleans up massive `.wfc` and `.dump` files to maintain an $O(1)$ storage footprint.

## Architecture Overview

PyAceMaker utilizes a 4-phase state machine orchestrated centrally, interacting dynamically with external simulation and training engines.

```mermaid
flowchart TD
    A[Initial Configuration] --> B(Orchestrator State Machine)

    subgraph Phase 1: Zero-Shot Distillation
    C1[Generate Combinatorial Structures] --> C2[MACEManager: Filter Uncertainty]
    C2 --> C3[Pacemaker: Baseline Delta Learning]
    end

    subgraph Phase 2: Validation
    V1[Validator: Phonon, Elastic, EOS Stress Test]
    end

    subgraph The Production Loop (Phases 3 & 4)
    E1[LAMMPS MD Engine] -- Continuous Run --> E2{Uncertainty Watchdog}
    E2 -- Halt Trigger --> P3_1[Intelligent Cutout & Passivation]
    P3_1 --> P3_2[QEDriver: Ground Truth DFT]
    P3_2 --> P4_1[MACE Finetune & Surrogate Gen]
    P4_1 --> P4_2[IncrementalTrainer: Delta Update]
    P4_2 -. Seamless Resume .-> E1
    end

    B --> C1
    C3 --> V1
    V1 --> E1
```

## Prerequisites

*   **Python**: >= 3.11
*   **Package Manager**: `uv` (recommended)
*   **DFT Code**: Quantum Espresso (`pw.x` in PATH)
*   **MLIP Frameworks**:
    *   Pacemaker (`pace_train` in PATH)
    *   MACE (`mace` python package or executable)
*   **MD Engine**: LAMMPS Python Interface (`lammps` package, compiled with `USER-PACE` support)

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker

# 2. Install dependencies using uv
uv sync

# 3. Setup configuration
cp .env.example .env
```

## Usage

1.  **Prepare Configuration**:
    Create a `config.yaml` leveraging the new Version 2.1.0 schemas.

    ```yaml
    project_name: "FePt_Alloy_NextGen"
    distillation:
        enable: true
        mace_model_path: "mace-mp-0-medium"
        uncertainty_threshold: 0.05
        sampling_structures_per_system: 1000
    loop_strategy:
        use_tiered_oracle: true
        incremental_update: true
        replay_buffer_size: 500
        baseline_potential_type: "LJ"
        thresholds:
            threshold_call_dft: 0.05
            threshold_add_train: 0.02
            smooth_steps: 3
    cutout:
        core_radius: 4.0
        buffer_radius: 3.0
        enable_pre_relaxation: true
        enable_passivation: true
        passivation_element: "H"
    # ... (DFT, MD, and Training settings remain similar to v1)
    ```

2.  **Run the Orchestrator**:

    ```bash
    # Run the full 4-Phase pipeline
    uv run pyacemaker --config config.yaml
    ```

## Development Workflow

This project enforces strict code quality using a 4-cycle development approach.

*   **Run Tests**:
    ```bash
    uv run pytest
    ```
*   **Run Linters & Type Checks**:
    ```bash
    uv run ruff check
    uv run mypy src
    ```
*   **Interactive Tutorials (UAT)**:
    Launch the User Acceptance Testing scenarios using Marimo:
    ```bash
    uv run marimo edit tutorials/UAT_AND_TUTORIAL.py
    ```

## Project Structure

```
src/pyacemaker/
├── core/
│   ├── base.py                 # Abstract base classes
│   ├── engine.py               # Seamless Resume MD wrappers
│   ├── loop.py                 # 4-Phase State Machine Orchestrator
│   ├── oracle.py               # TieredOracle & MACEManager
│   ├── trainer.py              # Incremental Delta Learning
│   └── state_manager.py        # Task-level HPC checkpointing
├── domain_models/
│   ├── config.py               # Pydantic schemas (Distillation, Cutout, etc.)
│   └── data.py                 # Core data structures (AtomStructure)
├── interfaces/                 # Wrappers for QE, LAMMPS, Pacemaker
├── utils/
│   ├── extraction.py           # Intelligent Cutout & Auto-passivation
│   └── cleanup.py              # HPC artifact management daemon
└── main.py                     # CLI Entry
```

## License

MIT License
