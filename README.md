# PyAceMaker

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Verified-brightgreen.svg)

**Next-Generation Adaptive Machine Learning Interatomic Potentials Orchestrator.**

PyAceMaker entirely revolutionizes active learning for molecular dynamics. By employing a "Hierarchical Distillation" architecture featuring foundation models like MACE and highly accurate DFT computations via Quantum ESPRESSO, it solves the critical challenges of long-timescale molecular dynamics simulations.

## Key Features

*   **Zero-Shot Distillation:** Generate incredibly robust baseline interatomic potentials using combinatorial structures and MACE foundation models strictly without any initial, expensive DFT calls.
*   **Two-Tier Uncertainty Thresholding:** Intelligently differentiates entirely between harmless transient thermal noise and true physical events via `TwoTierEvaluator`, completely avoiding unnecessary simulation pauses.
*   **Intelligent Cutout & Auto-Passivation:** Safely, mathematically extracts the precise epicenter of uncertainty and automatically passivates highly dangerous dangling bonds (e.g., smoothly adding fractional hydrogen) ensuring safe and remarkably reliable DFT calculations.
*   **Seamless MD Resume:** Features a robust Master-Slave inversion mechanism absolutely allowing the LAMMPS C++ engine to resume exactly, seamlessly from the halted step (perfectly preserving time, continuous coordinates, and momentum) rather than destructively resetting the entire MD loop.
*   **Incremental Delta Learning:** Intelligently mixes entirely newly generated AI surrogate data with a highly managed historical replay buffer of past interactions exactly to rapidly mathematically update potentials, fully mitigating O(N) computational bottlenecks and permanently preventing catastrophic forgetting.

## Architecture Overview

PyAceMaker employs an advanced state-machine driven orchestration model strictly using modern software design patterns. It integrates a foundational MACE AI oracle directly alongside Quantum ESPRESSO entirely to filter deep uncertainty, rigorously validating learned structural configurations strictly across several phases completely from local foundation fine-tuning entirely up to full scale, O(1) complex continuous MD resumption.

```mermaid
graph TD
    %% Subsystems
    subgraph Config & Schema
        CFG[Workflow Config / Pydantic]
    end

    subgraph Core Orchestrator
        ORCH[Main Orchestrator]
        STATE[State Manager / SQLite]
    end

    subgraph Phase 1: Distillation
        P1_GEN[Combinatorial Generator]
        P1_DIR[ActiveSet Selector]
        P1_ORACLE[MACEManager Oracle]
        P1_TRAIN[Pacemaker Trainer]
    end

    subgraph Phase 2: Validation
        P2_VAL[Validator Subsystem]
        P2_PHONON[Phonon / Elastic]
        P2_MINIMD[Miniature MD Test]
    end

    subgraph Phase 3 & 4: Active Learning Loop
        LAMMPS[LammpsEngine C++ Loop]
        EVAL[Two-Tier Evaluator]
        CUTOUT[Intelligent Cutout]
        PASSIVATE[Auto Passivation]
        DFT[DFTManager / QEDriver]
        SURR[Surrogate Generator]
        DELTA[Incremental Trainer]
    end

    %% Flow Phase 1
    CFG --> ORCH
    ORCH --> P1_GEN
    P1_GEN -- Structure Pool --> P1_DIR
    P1_DIR -- Reduced Set --> P1_ORACLE
    P1_ORACLE -- Confident Data --> P1_TRAIN
    P1_TRAIN -- base.yace --> P2_VAL

    %% Flow Phase 2
    P2_VAL --> P2_PHONON
    P2_VAL --> P2_MINIMD
    P2_MINIMD -- Success --> LAMMPS
    P2_MINIMD -- Fail --> P1_GEN

    %% Flow Phase 3 & 4
    LAMMPS -- Halt Signal --> EVAL
    EVAL -- Thermal Noise --> LAMMPS
    EVAL -- True Event --> CUTOUT
    CUTOUT -- Buffer Relax --> P1_ORACLE
    CUTOUT --> PASSIVATE
    PASSIVATE -- Clean Cluster --> DFT
    DFT -- Ground Truth --> SURR
    SURR -- Awakened MACE --> P1_ORACLE
    SURR -- Large Dataset --> DELTA
    DELTA -- Replay Buffer --> DELTA
    DELTA -- updated.yace --> LAMMPS

    %% Storage
    ORCH --> STATE
    DELTA --> STATE
```

## Prerequisites

*   **Python**: 3.11+
*   **Package Manager**: `uv`
*   **DFT Code**: Quantum Espresso (`pw.x` executable strictly within PATH)
*   **MLIP Trainer**: Pacemaker (`pace_train`, `pace_activeset` executables strictly within PATH)
*   **MD Engine**: LAMMPS Python Interface (`lammps` package strictly compiled with `USER-PACE` support)
*   **Containers**: Docker (strictly optional for deployment)

## Installation & Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/pyacemaker-org/pyacemaker.git
    cd pyacemaker
    ```

2.  Sync the environment precisely utilizing `uv`:
    ```bash
    uv sync
    ```

3.  Configure your environment securely:
    ```bash
    cp .env.example .env
    ```

## Usage

PyAceMaker uses a strictly validated `config.yaml` to rigidly dictate exact execution parameters (e.g., LoopStrategy, Cutouts, Thresholds). You can actively explore the highly powerful full capabilities completely through our beautifully interactive tutorial.

### Run Interactive Tutorial
View and safely execute the entirely immersive user test scenarios exactly within a highly interactive marimo notebook interface entirely strictly inside a highly secure Mock Mode:
```bash
uv run marimo edit tutorials/UAT_AND_TUTORIAL.py
```
Or execute strictly headlessly for CI verification:
```bash
uv run python tutorials/UAT_AND_TUTORIAL.py
```

### Production Execution
```bash
# Validate your highly complex configuration completely safely in dry-run mode
uv run pyacemaker --config config.yaml --dry-run

# Start the continuous massive active learning loop strictly with entirely hierarchical distillation fully enabled
uv run pyacemaker --config config.yaml
```

## Development Workflow

Our highly active continuous development workflow strictly emphasizes incredibly robust testing and exactly pristine code quality strictly utilizing modern Python linters.

*   **Run Linter & Code Formatter:**
    ```bash
    uv run ruff check .
    uv run ruff format .
    ```

*   **Run Strict Static Type Checking:**
    ```bash
    uv run mypy src/ tests/
    ```

*   **Run Automated Test Suite:**
    Execute the highly comprehensive unit, deep integration, and mocked E2E test suites precisely via `pytest` to strictly completely verify structural changes:
    ```bash
    uv run pytest
    ```

Development strictly follows an entirely planned 6-cycle implementation workflow specifically designed to carefully entirely layer complex functionality entirely without tightly coupling dependencies.

## Project Structure

```text
pyacemaker/
├── src/pyacemaker/
│   ├── core/               # Highly strict execution orchestration (Engine, Trainer, Oracle, Validation)
│   ├── domain_models/      # Strongly typed Pydantic data schemas, continuous workflows, and entirely strict configuration constraints
│   ├── interfaces/         # Robust external compute software driver adapters (LAMMPS, QE, Pacemaker)
│   ├── scenarios/          # Extremely complex "Grand Challenge" highly specialized workflow completely customized overrides
│   ├── utils/              # Specialized spatial tools strictly for entirely intelligent cluster extraction, chemical passivation, and geometry embeddings
│   └── main.py             # Main CLI application entrypoint strictly for execution
├── tests/                  # Highly robust isolated test suites completely explicitly ensuring architectural compliance
└── tutorials/              # Fully interactive Marimo notebooks entirely completely proving strict UAT highly explicit capabilities
```

## License

This strictly completely robust continuous orchestration completely platform is entirely licensed specifically under the permissive MIT License.
