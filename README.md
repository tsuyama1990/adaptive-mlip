# PyAceMaker

**Automated Machine Learning Interatomic Potential Generation Pipeline**

> **Minimal DFT Cost. Maximum ACE Accuracy.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

**PyAceMaker** is a state-of-the-art framework designed to automate the construction of high-accuracy Machine Learning Interatomic Potentials (MLIPs). By leveraging **Knowledge Distillation** from large foundation models (MACE), PyAceMaker drastically reduces the number of expensive DFT calculations required to train a robust potential.

The system orchestrates a 7-step workflow that starts from zero knowledge, explores the chemical space using uncertainty-driven active learning, creates a massive surrogate dataset via MACE-driven MD, and finally distills this knowledge into a fast, lightweight ACE (Atomic Cluster Expansion) potential with Delta Learning corrections.

## Key Features

*   **Foundation Model Distillation**: Uses pre-trained MACE models as a "Surrogate Oracle" to guide exploration and label massive datasets cheaply.
*   **Uncertainty-Based Active Learning**: Automatically identifies and calculates only the most informative structures (high uncertainty), minimizing DFT usage.
*   **Delta Learning**: Fine-tunes the final ACE potential on the sparse DFT data to correct systematic errors in the MACE surrogate, achieving first-principles accuracy.
*   **Zero-Config Automation**: Designed to run end-to-end with a simple YAML configuration file.
*   **Resilient Orchestration**: Supports job resumption, state persistence, and HPC integration.

## Architecture

PyAceMaker follows a "Hub-and-Spoke" architecture centered around an Orchestrator that manages the 7-Step Distillation Workflow.

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

## Prerequisites

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (Recommended) or `pip`
*   **Dependencies**: `mace-torch`, `pacemaker`, `ase`, `numpy`, `torch`
*   **Optional**: `vasp`, `quantum-espresso` (for Real Mode execution)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies**:
    We use `uv` for fast, reliable dependency management.
    ```bash
    uv sync
    ```

3.  **Activate the environment**:
    ```bash
    source .venv/bin/activate
    ```

## Usage

### Quick Start (Mock Mode)

To verify the installation and see the pipeline in action without needing DFT or heavy GPUs, run the SN2 Reaction tutorial in Mock Mode:

```bash
export PYACEMAKER_MODE=MOCK
python tutorials/UAT_AND_TUTORIAL.py
# Or use marimo for an interactive experience
marimo run tutorials/UAT_AND_TUTORIAL.py
```

### Production Run

1.  **Create a configuration file** (`config.yaml`):
    ```yaml
    work_dir: "./runs/experiment_01"
    distillation:
      enable_mace_distillation: true
      step1_direct_sampling:
        target_points: 100
    ```

2.  **Run the pipeline**:
    ```bash
    pyacemaker run --config config.yaml --all
    ```

3.  **Resume a stopped job**:
    ```bash
    pyacemaker run --config config.yaml --resume
    ```

## Development Workflow

This project is developed in 6 sequential cycles.

1.  **Cycle 01**: Core Framework & DIRECT Sampling
2.  **Cycle 02**: MACE Oracle & Active Learning
3.  **Cycle 03**: MACE Surrogate Loop
4.  **Cycle 04**: Surrogate Labeling & Base ACE Training
5.  **Cycle 05**: Delta Learning
6.  **Cycle 06**: Full Orchestration & Polish

### Running Tests

```bash
pytest tests/
```

### Linting & Code Quality

We enforce strict type checking and linting.

```bash
ruff check src tests
mypy src
```

## Project Structure

```text
pyacemaker/
├── config.yaml                 # Default configuration
├── pyproject.toml              # Dependencies and Tool Config
├── src/
│   └── pyacemaker/
│       ├── main.py             # CLI Entry point
│       ├── orchestrator.py     # Main workflow controller
│       ├── domain_models/      # Pydantic Data Models
│       ├── oracle/             # MACE/DFT Wrappers
│       ├── trainer/            # Pacemaker/MACE Training logic
│       └── structure_generator/# DIRECT/MD Sampling
└── tutorials/
    └── UAT_AND_TUTORIAL.py     # Interactive Tutorial
```

## License

MIT License. See `LICENSE` for details.
