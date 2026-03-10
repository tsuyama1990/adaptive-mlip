# PyAceMaker

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Verified-brightgreen.svg)

**Next-Generation Adaptive Machine Learning Interatomic Potentials Orchestrator.**

## Overview

**PyAceMaker** is an advanced orchestration platform designed to automate the construction of robust Machine Learning Interatomic Potentials (MLIPs). It orchestrates the entire active learning loop with a "Hierarchical Distillation" architecture to solve the critical challenges of long-timescale molecular dynamics simulations. PyAceMaker seamlessly pauses MD without losing time-continuity, repairs extracted clusters by auto-passivating dangling bonds, filters thermal noise using a two-tier uncertainty system, and incrementally updates potentials to prevent catastrophic forgetting.

## Key Features

*   **Zero-Shot Distillation:** Generate robust baseline potentials using combinatorial structures and MACE foundation models without initial expensive DFT calls.
*   **Two-Tier Uncertainty Thresholding:** Differentiates between harmless thermal noise and true physical events, avoiding unnecessary pauses.
*   **Intelligent Cutout & Auto-Passivation:** Safely extracts the epicenter of uncertainty and automatically passivates dangling bonds (e.g., adding fractional hydrogen) ensuring safe and reliable DFT calculations.
*   **Seamless MD Resume:** Features Master-Slave inversion allowing LAMMPS to resume exactly from the halted step (preserving time, coordinates, and velocities) rather than resetting the MD loop.
*   **Incremental Delta Learning:** Mixes newly generated surrogate data with a replay buffer of past interactions to rapidly update potentials, mitigating O(N) computational bottlenecks and catastrophic forgetting.

## Architecture Overview

PyAceMaker employs a state-machine driven orchestration model using modern software design patterns. It integrates a foundational MACE oracle alongside Quantum Espresso to filter uncertainty, validating learned configurations across several stages from local fine-tuning up to full scale, O(1) complex MD resumption.

```mermaid
graph TD
    %% Config and State
    CFG[Pydantic Config]
    STATE[(State DB / Files)]

    %% Orchestrator
    ORCH[Orchestrator]

    %% Generators
    GEN[Structure Generator]
    ACT[Active Set Selector]

    %% Oracles
    subgraph Oracles
        MACE[MACEManager]
        TIER[TieredOracle]
        DFT[DFTManager / QEDriver]
    end

    %% Training
    TRAIN[PacemakerTrainer]
    FINE[FinetuneManager]

    %% MD Engine
    ENG[LammpsEngine]
    VAL[Validator]

    %% Utilities
    CUT[Intelligent Cutout & Passivation]

    %% Phase 1
    CFG --> ORCH
    ORCH --> GEN
    GEN --> ACT
    ACT --> MACE
    MACE -- Confident Data --> TRAIN
    TRAIN -- Base Potential --> VAL

    %% Phase 3 & 4 (Active Loop)
    VAL -- Pass --> ENG
    ENG -- Uncertainty Halt --> TIER
    TIER -- High Uncertainty --> CUT
    CUT -- Safe Cluster --> DFT
    DFT -- Ground Truth --> FINE
    FINE -- Awakened MACE --> MACE
    MACE -- Surrogate Data --> TRAIN
    TRAIN -- Incremental Update --> ENG

    ORCH --> STATE
```

## Prerequisites

*   **Python**: 3.11+
*   **Package Manager**: `uv`
*   **DFT Code**: Quantum Espresso (`pw.x` executable in PATH)
*   **MLIP Trainer**: Pacemaker (`pace_train`, `pace_activeset` executables in PATH)
*   **MD Engine**: LAMMPS Python Interface (`lammps` package, with `USER-PACE` support)

## Installation & Setup

1. Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
```

2. Initialize the project using `uv` to install all dependencies:
```bash
uv sync
```

3. Set up your environment variables (if applicable):
```bash
cp .env.example .env
```

## Usage

PyAceMaker uses a `config.yaml` to dictate execution (LoopStrategy, Cutouts, etc). You can explore the full capabilities through our interactive tutorial.

### Run Interactive Tutorial
View and execute the user scenarios in a notebook interface:
```bash
uv run marimo edit tutorials/UAT_AND_TUTORIAL.py
```

### Production Execution
```bash
# Validate your configuration in dry-run mode
uv run pyacemaker --config config.yaml --dry-run

# Start the active learning loop with hierarchical distillation enabled
uv run pyacemaker --config config.yaml
```

## Development Workflow

Active development emphasizes robust testing and code quality based on the specified cycles:

*   **Run Tests:**
    Execute the unit, integration, and E2E test suites with `pytest` utilizing `uv` to ensure proper environment resolution.
    ```bash
    uv run pytest tests/
    ```

*   **Run Linter:**
    Strict rules (e.g. cyclomatic complexity, explicit types) are enforced.
    ```bash
    uv run ruff check .
    ```

*   **Run Formatter:**
    ```bash
    uv run ruff format .
    ```

*   **Run Type Checking:**
    Strict type checking on the source code.
    ```bash
    uv run mypy src/
    ```

## Project Structure

```text
src/pyacemaker/
├── core/               # Execution orchestration (Engine, Trainer, Oracle, Validation)
├── domain_models/      # Pydantic data schemas, workflows, and strict configuration constraints
├── interfaces/         # External compute driver adapters (LAMMPS, QE, Pacemaker)
├── scenarios/          # Complex "Grand Challenge" specialized workflow overrides
├── utils/              # Tools for cluster extraction, passivation, geometry embeddings
└── main.py             # CLI application entrypoint
```

## License

This project is licensed under the MIT License.