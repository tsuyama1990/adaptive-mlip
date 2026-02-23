# PYACEMAKER

**High-Efficiency MLIP Construction & Operation System**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

PYACEMAKER is an autonomous system designed to democratise the creation of Machine Learning Interatomic Potentials (MLIPs). By orchestrating the **Pacemaker** engine within a self-driving Active Learning loop, it allows researchers to generate State-of-the-Art potentials for complex alloys and interfaces with a "Zero-Config" workflow—no coding required.

---

## 🚀 Key Features

*   **Zero-Config Automation**: Launch a full active learning campaign with a single YAML file. The system automatically infers hyperparameters based on physical properties.
*   **Physics-Informed Robustness**: Guarantees physical safety (no atomic fusion) by enforcing a Delta Learning strategy with LJ/ZBL baselines, ensuring simulations never crash due to non-physical forces.
*   **Data Efficiency**: Utilizes D-Optimality (Active Set selection) and Periodic Embedding to achieve DFT accuracy with 90% fewer calculations than random sampling.
*   **Self-Healing Oracle**: Automatically recovers from Quantum Espresso convergence failures by adjusting mixing betas and smearing parameters on the fly.
*   **Multi-Scale Capability**: Seamlessly bridges the gap between Molecular Dynamics (LAMMPS) and Adaptive Kinetic Monte Carlo (EON) for long-timescale simulations.

---

## 🏗 Architecture Overview

The system follows a cyclic "Explore-Label-Train-Run" architecture managed by a central Orchestrator.

```mermaid
graph TD
    User[User] -->|config.yaml| ORC[Orchestrator]
    ORC -->|1. Explore| SG[Structure Generator]
    SG -->|Candidates| ORC
    ORC -->|2. Compute| DFT[Oracle (QE)]
    DFT -->|Data| ORC
    ORC -->|3. Train| PM[Trainer (Pacemaker)]
    PM -->|potential.yace| ORC
    ORC -->|4. Run| MD[Engine (LAMMPS)]
    MD -- High Uncertainty --> ORC
```

## 🛠 Prerequisites

*   **Python**: 3.11 or higher
*   **Quantum Espresso**: `pw.x` must be in your PATH (for Oracle)
*   **LAMMPS**: Must be installed with `USER-PACE` package
*   **Pacemaker**: `pace_train`, `pace_activeset` tools
*   **EON**: (Optional) For kMC simulations

## 📦 Installation & Setup

We recommend using `uv` or `poetry` for dependency management, but `pip` works as well.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    ```bash
    pip install .
    # OR if using uv
    uv sync
    ```

3.  **Environment Setup**
    Copy the example configuration and adjust for your HPC environment.
    ```bash
    cp config.example.yaml config.yaml
    ```

## ⚡ Usage

### Quick Start (FePt Alloy)

1.  **Prepare Configuration**
    Create a `config.yaml` file:
    ```yaml
    project_name: "FePt_Test"
    elements: ["Fe", "Pt"]
    temperature: 1000
    ```

2.  **Run the Orchestrator**
    ```bash
    python -m pyacemaker.main --config config.yaml
    ```

3.  **Monitor Progress**
    Tail the logs to see the cycle in action:
    ```bash
    tail -f pyacemaker.log
    ```

### Running Tutorials
We provide interactive Marimo tutorials for validation.
```bash
marimo edit tutorials/UAT_AND_TUTORIAL.py
```

## 💻 Development Workflow

This project uses a cycle-based development approach (Cycles 01-08).

*   **Testing**: Run the full test suite.
    ```bash
    pytest
    ```
*   **Linting**: Ensure code quality with Ruff and Mypy.
    ```bash
    ruff check .
    mypy src
    ```

## 📂 Project Structure

```ascii
pyacemaker/
├── dev_documents/          # Specs & Architecture
│   ├── system_prompts/     # Cycle Definitions (01-08)
│   └── SYSTEM_ARCHITECTURE.md
├── src/
│   └── pyacemaker/
│       ├── core/           # Core Logic (Generator, Oracle, Trainer)
│       ├── interfaces/     # Drivers (QE, LAMMPS, EON)
│       └── orchestrator.py # Main Loop
├── tests/                  # Pytest Suite
└── tutorials/              # Marimo Notebooks
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
