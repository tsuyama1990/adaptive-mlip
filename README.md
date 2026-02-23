# PYACEMAKER

**High-Efficiency MLIP Construction & Operation System**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

PYACEMAKER is an autonomous system designed to democratise the creation of Machine Learning Interatomic Potentials (MLIPs). By orchestrating the **Pacemaker** engine within a self-driving Active Learning loop, it allows researchers to generate State-of-the-Art potentials for complex alloys and interfaces with a "Zero-Config" workflow.

**Current Status**: Configuration & Orchestration System Verified

---

## 🚀 Key Features

*   **Robust Configuration**: Utilizes **Pydantic V2** for strict schema validation, ensuring all inputs (temperatures, cutoffs, paths) are physically valid before execution.
*   **Orchestration Core**: Centralized state machine designed to manage the "Explore-Label-Train-Run" lifecycle.
*   **Structured Logging**: Automatic setup of console and file logging with rotation policies.
*   **Modular Architecture**: Clean separation of concerns with Abstract Base Classes for Generator, Oracle, Trainer, and Engine.
*   **Zero-Config Automation**: (Planned) Launch a full active learning campaign with a single YAML file.

---

## 🛠 Prerequisites

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (recommended) or `pip`

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    Using `uv`:
    ```bash
    uv sync
    ```
    Using `pip`:
    ```bash
    pip install .
    ```

## ⚡ Usage

### 1. Create a Configuration File
Create a `config.yaml` file with the required sections:

```yaml
project_name: "FePt_Optimization"

structure:
  elements: ["Fe", "Pt"]
  supercell_size: [2, 2, 2]

dft:
  code: "quantum_espresso"
  encut: 500.0
  kpoints_density: 0.04

training:
  potential_type: "ace"
  cutoff_radius: 5.0
  max_basis_size: 500

md:
  temperature: 1000.0
  n_steps: 1000

workflow:
  max_iterations: 10
  convergence_energy: 0.001
```

### 2. Validate Configuration (Dry Run)
Check if your configuration is valid without running any simulations:

```bash
uv run python -m pyacemaker.main --config config.yaml --dry-run
```
*Output: "Configuration loaded successfully."*

### 3. Run the Orchestrator
Start the active learning loop (currently a skeleton loop for Cycle 01):

```bash
uv run python -m pyacemaker.main --config config.yaml
```

## 🏗 Architecture & File Structure

```ascii
pyacemaker/
├── pyproject.toml              # Dependencies & Settings
├── src/
│   └── pyacemaker/
│       ├── domain_models/      # Pydantic Schemas (Config, Structure, DFT, etc.)
│       ├── core/               # Abstract Base Classes (Generator, Oracle, etc.)
│       ├── utils/              # Utilities (IO, YAML parsing)
│       ├── logger.py           # Logging setup
│       ├── orchestrator.py     # Main Logic
│       └── main.py             # CLI Entry Point
└── tests/                      # Unit, E2E, and UAT tests
```

## 💻 Development

*   **Testing**:
    ```bash
    uv run pytest
    ```
*   **Linting**:
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

## 📄 License

This project is licensed under the MIT License.
