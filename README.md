# PyAceMaker: NextGen Hierarchical Distillation Architecture

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

PyAceMaker is an advanced Active Learning orchestration framework designed to generate highly accurate Machine Learning Interatomic Potentials (MLIPs). By leveraging Foundation Models (like MACE) and a novel Hierarchical Distillation approach, it enables unbroken, million-atom Molecular Dynamics (MD) simulations that seamlessly bridge the gap between empirical speed and ab-initio accuracy.

## Key Features

1. **Zero-Shot Distillation Baseline:** Automatically generate a robust initial potential utilizing Foundation Models (e.g., MACE-MP-0) without requiring a single expensive DFT calculation.
2. **Seamless Time-Continuous MD:** Built on a Master-Slave Inversion paradigm (via LAMMPS `fix python/invoke`), MD simulations pause, update their underlying potential, and resume from the exact phase space without breaking continuity.
3. **Two-Tier Evaluator (Noise Filtering):** Intelligently separates uncertainty thresholds to prevent non-critical thermal noise from triggering expensive ab-initio calculations.
4. **Intelligent Cutout & Auto-Passivation:** When an anomaly is detected, the system safely extracts a localized cluster, pre-relaxes the boundary using MACE, and passivates dangling bonds before sending it to the Quantum ESPRESSO DFT solver.
5. **O(1) Incremental Delta Learning:** Eradicates catastrophic forgetting and computational bottlenecks by utilizing replay buffers and updating only the necessary weights via difference learning, keeping training time constant regardless of history size.

## Architecture Overview

PyAceMaker separates the physically intensive MD engine from the machine learning orchestration and Ab-Initio solvers, tying them together through strictly validated Pydantic Domain Models.

```mermaid
graph TD
    A[Initialization & Zero-Shot Distillation] --> B(MD Engine loop)
    B --> C{Two-Tier Evaluator}
    C -- "Thermal Noise (Ignore)" --> B
    C -- "Critical Anomaly (Halt)" --> D[Intelligent Cutout & Passivation]
    D --> E{Tiered Oracle}
    E -- "MACE Confident" --> F[Surrogate Data Generation]
    E -- "High Uncertainty" --> G[Quantum ESPRESSO (DFT)]
    G --> F
    F --> H[Incremental Delta Learning]
    H --> B
```

## Prerequisites

- **Python 3.12+**
- `uv` package manager
- LAMMPS (compiled with Python support if using `fix python/invoke`)
- Quantum ESPRESSO (for DFT calculations)
- CUDA-compatible GPU (Highly recommended for MACE inference and Pacemaker training)

## Installation & Setup

We recommend using `uv` for fast dependency resolution and virtual environment management.

```bash
# Clone the repository
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker

# Sync dependencies using uv
uv sync

# Set up your environment variables (e.g., HPC scheduler prefixes)
cp .env.example .env
```

## Usage

### Quick Start (Tutorial Mode)

The best way to understand PyAceMaker is through our interactive, single-file Marimo notebook. It includes a "Mock Mode" that allows you to run the entire loop on your laptop without needing external DFT binaries.

```bash
uv run python tutorials/UAT_AND_TUTORIAL.py
```

### Production Run

To start a full production active learning loop based on your configuration file:

```bash
uv run pyacemaker --config config.yaml
```

## Development Workflow

PyAceMaker adheres to strict code quality standards, enforced via `ruff` and `mypy`.

### Running Tests

Execute the test suite with coverage reporting:
```bash
uv run pytest tests/
```

### Running Linters and Type Checks

Ensure your code meets the project's formatting and type-safety requirements before committing:
```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Project Structure

```text
pyacemaker/
├── pyproject.toml
├── src/
│   └── pyacemaker/
│       ├── core/           # Orchestrator, Engine interfaces, Trainer, Oracle
│       ├── domain_models/  # Pydantic Schemas (Config, MD, Structure)
│       ├── interfaces/     # Secure subprocess execution (e.g., QE Driver)
│       └── utils/          # Intelligent extraction, embedding, passivation
├── tests/                  # Pytest unit and integration tests
├── tutorials/              # Marimo UAT and usage tutorials
└── dev_documents/          # Architecture specs and Gherkin UAT definitions
```

## License

This project is licensed under the MIT License.

## License

This project is licensed under the MIT License.
