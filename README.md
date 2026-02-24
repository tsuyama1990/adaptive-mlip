# PyAceMaker

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Cycle%2007%20Verified-brightgreen.svg)

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
    *   **Self-Healing**: Automatically retries failed calculations with adjusted parameters.
    *   **Security**: Prevents path traversal and validates input configuration.
*   **Potential Training (Pacemaker)**:
    *   **Delta Learning**: Fits the difference between DFT and a physics-based baseline (LJ/ZBL).
    *   **Active Set Optimization**: Uses D-optimality (MaxVol) to select the most informative structures.
    *   **Automated Configuration**: Generates optimal `input.yaml` for Pacemaker based on dataset composition.
*   **Molecular Dynamics (MD) Engine**:
    *   Integrated LAMMPS driver for NPT/NVT simulations.
    *   **Hybrid Potentials**: Overlays ACE with ZBL/LJ for safety during high-energy events.
    *   **Uncertainty Watchdog**: Automatically halts simulations when the extrapolation grade ($\gamma$) exceeds a safe threshold.
*   **Validation & QA (The Guardian)**:
    *   **Phonon Stability**: Calculates phonon band structures using Phonopy to detect imaginary modes (dynamical instability).
    *   **Elastic Stability**: Computes elastic constants ($C_{ij}$) and checks Born stability criteria (mechanical stability).
    *   **Automated Reporting**: Generates HTML reports summarizing validation results.
*   **Scalability**:
    *   **Streaming Data Processing**: Handles large datasets with O(1) memory usage.
    *   **Resume Capability**: Checkpoints state to `state.json`, allowing workflows to pause and resume from the exact iteration.

## Requirements

*   **Python**: >= 3.11
*   **DFT Code**: Quantum Espresso (`pw.x` executable in PATH)
*   **MLIP Trainer**: Pacemaker (`pace_train`, `pace_activeset` executables in PATH)
*   **MD Engine**: LAMMPS Python Interface (`lammps` package, with `USER-PACE` support)
*   **Validation**: Phonopy (installed automatically via `uv sync`)

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
        code: "qe"
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
        delta_learning: true
        active_set_optimization: true
    md:
        temperature: 1000.0
        pressure: 0.0
        timestep: 0.001
        n_steps: 5000
    workflow:
        max_iterations: 10
        state_file_path: "state.json"
        data_dir: "data"
        active_learning_dir: "active_learning"
        potentials_dir: "potentials"
    validation:
        enabled: true
        phonon:
            supercell_size: [2, 2, 2]
            displacement: 0.01
        elastic:
            strain_magnitude: 0.01
    logging:
        level: "INFO"
        log_file: "pyacemaker.log"
    ```

2.  **Run PyAceMaker**:

    ```bash
    # Dry run to validate config
    uv run python src/pyacemaker/main.py --config config.yaml --dry-run

    # Start the active learning loop
    uv run python src/pyacemaker/main.py --config config.yaml
    ```

## Architecture

```
src/pyacemaker/
├── core/               # Core business logic (Generator, Oracle, Trainer, Validator)
├── domain_models/      # Pydantic data schemas and validation
├── interfaces/         # External code drivers (Quantum Espresso, LAMMPS)
├── utils/              # Helper functions (I/O, perturbations, phonons, elastic)
├── factory.py          # Dependency injection
├── orchestrator.py     # Workflow state machine
└── main.py             # CLI entry point
```

## Roadmap

*   [ ] Support for VASP DFT code.
*   [ ] Integration with MACE potentials.
*   [ ] Advanced active learning strategies (Query by Committee).
