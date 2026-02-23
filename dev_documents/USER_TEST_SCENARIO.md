# User Test Scenario: Fe/Pt Deposition on MgO

## 1. Grand Challenge
**Goal**: Simulate the deposition of Iron (Fe) and Platinum (Pt) atoms onto a Magnesium Oxide (MgO) (001) substrate, observe the nucleation of clusters, and visualize the L10 ordering process using a combination of Molecular Dynamics (MD) and Adaptive Kinetic Monte Carlo (aKMC).

## 2. Tutorial Strategy

To ensure this complex scientific workflow is accessible and verifiable, we will implement a "Dual-Mode" strategy.

### 2.1. Modes of Operation
-   **Mock Mode (CI/CD)**: Designed for GitHub Actions or users without high-performance computing resources.
    -   **Substrate**: Tiny $2 \times 2 \times 1$ supercell.
    -   **Deposition**: 5 atoms only.
    -   **MD**: 100 steps (dummy run).
    -   **Oracle/Trainer**: Mocked (returns pre-calculated potentials/energies).
    -   **Execution Time**: < 5 minutes.
-   **Real Mode (Production)**: Designed for workstations with 16+ cores.
    -   **Substrate**: Large $10 \times 10 \times 4$ slab.
    -   **Deposition**: 500+ atoms.
    -   **MD**: 1,000,000 steps.
    -   **Oracle/Trainer**: Real Quantum Espresso and Pacemaker execution.
    -   **Execution Time**: Hours/Days.

### 2.2. Technology Stack
-   **Marimo**: We will use a single Marimo notebook `tutorials/UAT_AND_TUTORIAL.py` as the executable documentation. This allows interactive visualization and code execution in a reactive environment.
-   **ASE**: Atomic Simulation Environment for structure manipulation.
-   **PyVista/Matplotlib**: For in-notebook visualization.

## 3. Tutorial Plan (The Marimo File)

The `tutorials/UAT_AND_TUTORIAL.py` will contain the following sections:

### Section 1: Setup & Initialization
-   Import `pyacemaker`.
-   Detect environment (`CI=true` or `false`) to set simulation parameters.
-   Initialize `Orchestrator`.

### Section 2: Phase 1 - Divide & Conquer Training (Active Learning)
-   **Step A**: Train MgO bulk & surface potential.
-   **Step B**: Train Fe-Pt alloy potential (L10 phase).
-   **Step C**: Train Interface potential (Fe/Pt on MgO).
-   *Visualization*: Show the training error convergence plot.

### Section 3: Phase 2 - Dynamic Deposition (MD)
-   Load the trained hybrid potential.
-   Set up LAMMPS `fix deposit`.
-   Run MD simulation (Mock or Real).
-   *Visualization*: Interactive 3D view of atoms landing on the surface.

### Section 4: Phase 3 - Long-Term Ordering (aKMC)
-   Take the final MD snapshot (disordered cluster).
-   Bridge to EON for aKMC (or mock results).
-   Observe L10 ordering (chemically ordered layers).
-   *Visualization*: Show the "Order Parameter" vs Time graph.

## 4. Validation Criteria
-   **Crash-Free**: The notebook must run top-to-bottom without error in CI mode.
-   **Physics Check**:
    -   Potential energy must be negative.
    -   No atoms should be closer than 1.5 Å (Core repulsion check).
-   **Artifacts**:
    -   `potential.yace` file created.
    -   `trajectory.xyz` file created.
    -   `report.html` generated.
