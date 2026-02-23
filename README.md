# PyAceMaker

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Verified-brightgreen.svg)

**Adaptive machine learning interatomic potentials construction orchestrator.**

## Overview

**PyAceMaker** is an automated workflow tool designed to construct robust Machine Learning Interatomic Potentials (MLIPs). It orchestrates the entire active learning loop: generating candidate structures, running DFT calculations (via Quantum Espresso), training potentials (e.g., ACE), and validating them through MD simulations.

### Why PyAceMaker?
Constructing MLIPs manually is tedious and error-prone. PyAceMaker automates the "Active Learning" cycle, ensuring that your potential is trained on the most relevant structures—those where the current model is uncertain—thereby maximizing accuracy while minimizing expensive DFT calls.

## Features

*   **Automated Workflow Orchestration**: Manages the loop of exploration, labeling, training, and validation.
*   **Structure Exploration**:
    *   **Cold Start**: Initial structure generation using M3GNet (or database lookup).
    *   **Perturbation Policies**:
        *   **Random Rattle**: Displaces atoms to sample local energy minima.
        *   **Strain**: Applies volume and shear strain to sample elastic response.
        *   **Defects**: Introduces vacancies to train for off-stoichiometry robustness.
*   **DFT Management**:
    *   Automated Quantum Espresso execution via ASE.
    *   **Self-Healing**: Automatically retries failed calculations with adjusted parameters (e.g., mixing beta, smearing).
    *   **Security**: Prevents path traversal and validates input configuration.
*   **Scalability**:
    *   **Streaming Data Processing**: Handles large datasets with O(1) memory usage.
    *   **Resume Capability**: Checkpoints state to JSON, allowing workflows to pause and resume.

## Requirements

*   **Python**: >= 3.11
*   **DFT Code**: Quantum Espresso (`pw.x` executable in PATH)
*   **MLIP Trainer**: (e.g., `pace`, `mace`, etc. depending on plugin)

## Installation

```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
uv sync
```

## Usage

1.  **Prepare Configuration**:
    Create a `config.yaml` file defining your project parameters.

    ```yaml
    project_name: "FePt_Alloy"
    structure:
        elements: ["Fe", "Pt"]
        supercell_size: [2, 2, 2]
        policy_name: "random_rattle"
        rattle_stdev: 0.1
    dft:
        code: "quantum_espresso"
        functional: "PBE"
        kpoints_density: 0.04
        encut: 500.0
        pseudopotentials:
            Fe: "Fe.pbe-n-kjpaw_psl.1.0.0.UPF"
            Pt: "Pt.pbe-n-kjpaw_psl.1.0.0.UPF"
    training:
        potential_type: "ace"
        cutoff_radius: 5.0
        max_basis_size: 500
    md:
        temperature: 1000.0
        pressure: 0.0
        timestep: 0.001
        n_steps: 5000
    workflow:
        max_iterations: 10
        checkpoint_interval: 1
    ```

2.  **Run PyAceMaker**:

    ```bash
    # Dry run to validate config
    uv run pyacemaker --config config.yaml --dry-run

    # Start the active learning loop
    uv run pyacemaker --config config.yaml
    ```

## Architecture

```
src/pyacemaker/
├── core/               # Core business logic (Generator, Oracle, Trainer)
├── domain_models/      # Pydantic data schemas and validation
├── interfaces/         # External code drivers (Quantum Espresso)
├── utils/              # Helper functions (I/O, perturbations)
├── factory.py          # Dependency injection
├── orchestrator.py     # Workflow state machine
└── main.py             # CLI entry point
```
