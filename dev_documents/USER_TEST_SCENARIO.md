# User Test Scenario: Fe/Pt Deposition on MgO

## 1. Grand Challenge Overview
**Goal**: Simulate the deposition of Iron (Fe) and Platinum (Pt) atoms onto a Magnesium Oxide (MgO) (001) substrate, observe the nucleation of clusters, and visualize the L10 ordering process using a combination of Molecular Dynamics (MD) and Adaptive Kinetic Monte Carlo (aKMC).

This scenario serves as the ultimate acceptance test for the **PYACEMAKER** system, validating the integration of all 8 development cycles.

## 2. Tutorial Strategy: "Dual-Mode" Execution

To ensure this complex scientific workflow is accessible for verification (CI/CD) and powerful enough for research, we implement a "Dual-Mode" strategy.

### 2.1. Modes of Operation
*   **Mock Mode (CI/CD & Quick Start)**:
    *   **Purpose**: Verify software logic without heavy computation.
    *   **Substrate**: Tiny $2 \times 2 \times 1$ supercell.
    *   **Deposition**: 5 atoms only.
    *   **MD**: 100 steps (dummy run).
    *   **Oracle/Trainer**: Mocked (returns pre-calculated energies/potentials).
    *   **Execution Time**: < 5 minutes.
    *   **Hardware**: Standard Laptop / GitHub Actions runner.

*   **Real Mode (Production Research)**:
    *   **Purpose**: Generate publishable scientific data.
    *   **Substrate**: Large $10 \times 10 \times 4$ slab.
    *   **Deposition**: 500+ atoms.
    *   **MD**: 1,000,000 steps.
    *   **Oracle/Trainer**: Real Quantum Espresso and Pacemaker execution.
    *   **Execution Time**: Hours/Days.
    *   **Hardware**: Workstation (16+ cores) or HPC.

### 2.2. Technology Stack
*   **Marimo**: We will use a single Marimo notebook `tutorials/UAT_AND_TUTORIAL.py` as the executable documentation. This allows interactive visualization and code execution in a reactive environment.
*   **ASE**: Atomic Simulation Environment for structure manipulation.
*   **PyVista/Matplotlib**: For in-notebook visualization.

## 3. Tutorial Plan (The Marimo File)

The `tutorials/UAT_AND_TUTORIAL.py` will contain the following sections, mapping to the development cycles:

### Section 1: Setup & Initialization (Cycle 01)
*   Import `pyacemaker`.
*   Detect environment (`CI=true` or `false`) to set simulation parameters.
*   Initialize `Orchestrator` with the appropriate `config.yaml`.
*   *Validation*: Check that configuration loads correctly.

### Section 2: Phase 1 - Divide & Conquer Training (Cycles 02-04)
*   **Step A (Oracle)**: Define the chemical system (Fe, Pt, Mg, O).
*   **Step B (Generator)**: Generate random bulk and surface structures (mocked in CI).
*   **Step C (Trainer)**: Train the Fe-Pt alloy potential (L10 phase) and MgO potential.
*   *Visualization*: Show the training error convergence plot (RMSE vs Epoch).

### Section 3: Phase 2 - Dynamic Deposition (Cycles 05-06)
*   **Step A (Engine)**: Load the trained hybrid potential.
*   **Step B (Orchestrator)**: Run the Active Learning Loop.
    *   Deposit atoms.
    *   Detect high uncertainty (simulated in Mock).
    *   Trigger refinement.
*   *Visualization*: Interactive 3D view of atoms landing on the surface (using `pyvista`).

### Section 4: Phase 3 - Quality Assurance (Cycle 07)
*   **Step A (Validator)**: Calculate phonon band structure for the L10-FePt phase.
*   *Validation*: Assert no imaginary frequencies (in Real mode) or check report generation (in Mock mode).

### Section 5: Phase 4 - Long-Term Ordering (Cycle 08)
*   **Step A (Expander)**: Take the final MD snapshot (disordered cluster).
*   **Step B (EON)**: Bridge to EON for aKMC.
*   **Step C (Result)**: Observe L10 ordering (chemically ordered layers).
*   *Visualization*: Show the "Order Parameter" vs Time graph.

## 4. Validation Criteria
*   **Crash-Free**: The notebook must run top-to-bottom without error in CI mode.
*   **Physics Check**:
    *   Potential energy must be negative.
    *   No atoms should be closer than 1.5 Å (Core repulsion check).
*   **Artifacts**:
    *   `potential.yace` file created.
    *   `trajectory.xyz` file created.
    *   `validation_report.html` generated.
