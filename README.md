# PyAceMaker
![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**Automated MLIP generation pipeline using MACE Knowledge Distillation.**

## Overview
**PyAceMaker** is an orchestration framework designed to automate the lifecycle of Machine Learning Interatomic Potential (MLIP) development. It streamlines the complex process of data generation, active learning, and potential training into a robust, self-healing pipeline.

**Why?** Developing MLIPs typically involves manual, error-prone steps: running DFT, training potentials, running MD, checking for failure, and repeating. PyAceMaker automates this "Knowledge Distillation" loop, significantly reducing time-to-solution for materials discovery.

## Features
*   **Automated Workflow Orchestration**: Manages the 7-step active learning cycle autonomously.
*   **DIRECT Sampling (Step 1)**:  Generates diverse initial structures using entropy maximization (Random Packing) to jumpstart the learning process without manual selection.
*   **Robust Configuration**: Uses strict schema validation (Pydantic) to prevent runtime errors due to invalid settings.
*   **Self-Healing Oracle**: (Planned) DFT interface with automatic error recovery.
*   **Mock Oracle**: Includes a built-in mock oracle (Lennard-Jones) for rapid testing and pipeline verification without expensive DFT calculations.

## Requirements
*   Python 3.11+
*   `uv` (Universal Python Package Manager) recommended for dependency management.
*   External Dependencies (Optional for Cycle 01):
    *   LAMMPS (for MD engine)
    *   Pacemaker (for training)
    *   Quantum Espresso (for DFT)

## Installation

```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
uv sync
```

## Usage

### 1. Initialize Workspace
Create a new workspace from a configuration file.

```bash
# Create a sample config (or use existing)
# Edit config.yaml to set your parameters
uv run pyacemaker init --config config.yaml
```

### 2. Run DIRECT Sampling (Step 1)
Execute the first step of the pipeline to generate initial structures.

```bash
uv run pyacemaker run --step 1
```

## Architecture
```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/           # Core logic (Engine, Oracle, Trainer)
│       ├── domain_models/  # Pydantic data schemas
│       ├── structure_generator/ # Sampling algorithms
│       ├── orchestrator.py # Main workflow controller
│       └── main.py         # CLI entry point
├── tests/                  # Unit and Integration tests
└── config.yaml             # Configuration template
```

## Roadmap
*   **Cycle 02**: Active Learning Loop (MD Exploration & Labeling).
*   **Cycle 03**: Training Integration (Pacemaker).
*   **Cycle 04**: Advanced Validation (Phonons, Elasticity).
*   **Cycle 05**: Adaptive Strategies.
*   **Cycle 06**: Production Hardening.
