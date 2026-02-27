# pyacemaker: Adaptive Machine Learning Interatomic Potentials

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

**Scalable, automated workflow for training MACE interatomic potentials.**

## Overview

### What is pyacemaker?
`pyacemaker` is an orchestration framework designed to automate the lifecycle of Machine Learning Interatomic Potentials (MLIPs). It manages the entire process: from initial structure generation and active learning loops to training and validation. It leverages **MACE** technology for state-of-the-art accuracy and efficiency.

### Why use it?
Training robust ML potentials requires complex iterative cycles of:
1.  Exploring chemical space.
2.  Running expensive DFT calculations (Oracle).
3.  Training models.
4.  Validating stability.

`pyacemaker` automates this "Active Learning" loop, ensuring:
*   **Reproducibility:** Every step is configured via code.
*   **Scalability:** Streaming architecture handles large datasets with O(1) memory usage.
*   **Resilience:** Self-healing DFT workflows recover from common convergence errors.
*   **Intelligence:** Uses entropy maximization (DIRECT sampling) and uncertainty quantification to minimize expensive data labeling.

## Features

*   **Core Data Models (Cycle 01):** Robust Pydantic-based schemas (`AtomStructure`) ensuring strict data validation and provenance tracking.
*   **Smart Sampling (Cycle 02):**
    *   **DIRECT Sampling:** Maximizes descriptor space entropy (using SOAP/ACE) to generate diverse initial datasets.
    *   **MACE Active Learning:** Uses MACE model uncertainty to intelligently select the most informative structures for labeling.
*   **Mock Oracle:** Built-in Mock Oracle for testing pipelines without expensive DFT codes.
*   **Configurable Workflow:** YAML-based configuration for all simulation parameters.
*   **Streaming I/O:** Efficient handling of large trajectory files (XYZ/LAMMPS) to prevent memory overflows.

## Requirements

*   Python 3.11 or higher
*   `uv` (recommended) or `pip`
*   LAMMPS (optional, for MD)
*   Quantum Espresso (optional, for DFT)
*   `mace-torch` and `dscribe` (automatically installed)

## Installation

```bash
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker
uv sync
```

## Usage

### Basic Execution

To run the orchestrator with a configuration file:

```bash
uv run pyacemaker --config config.yaml
```

### Configuration Example (`config.yaml`)

```yaml
project_name: "CuZr_ActiveLearning"

structure:
  elements: ["Cu", "Zr"]
  supercell_size: [2, 2, 2]

distillation:
  enable_mace_distillation: true
  step1_direct_sampling:
    target_points: 100
    descriptor:
      method: "soap"
      species: ["Cu", "Zr"]
      r_cut: 5.0
      n_max: 8
      l_max: 6
      sigma: 0.5
  step2_active_learning:
    uncertainty_threshold: 0.05
    n_active: 20
  step3_mace_finetune:
    base_model: "MACE-MP-0"

dft:
  code: "qe" # or "mock"
  functional: "PBE"
  kpoints_density: 0.04
  encut: 500.0
  pseudopotentials:
    Cu: "Cu.UPF"
    Zr: "Zr.UPF"

training:
  potential_type: "ace"
  cutoff_radius: 5.0
  max_basis_size: 500
  output_filename: "potential.yace"

md:
  temperature: 1000.0
  pressure: 0.0
  timestep: 0.002
  n_steps: 10000

workflow:
  max_iterations: 5
  data_dir: "data"
```

## Architecture

```text
src/pyacemaker/
├── core/               # Core logic (Orchestrator, Generator, Oracle, Trainer)
├── domain_models/      # Pydantic data schemas (AtomStructure, Configs)
├── modules/            # Concrete implementations (MockOracle, DirectSampler, MaceOracle)
├── interfaces/         # Drivers for external codes (LAMMPS, QE)
└── utils/              # Helper functions (I/O, Math, Descriptors)
```

## Roadmap

*   **Cycle 01:** Core Infrastructure & Data Models (Completed)
*   **Cycle 02:** Smart Sampling & Active Learning (Completed)
*   **Cycle 03:** Surrogate Refinement
*   **Cycle 04:** Distillation Phase
*   **Cycle 05:** Delta Learning
*   **Cycle 06:** Integration & User Experience
