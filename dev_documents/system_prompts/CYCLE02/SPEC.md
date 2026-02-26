# Cycle 02 Specification: The Oracle (DFT Automation)

## 1. Summary
This cycle implements the core **Oracle** module, responsible for generating ground-truth data using Density Functional Theory (DFT). The focus is on automating **Quantum Espresso (QE)** calculations through the `ase` (Atomic Simulation Environment) interface. A critical feature is the "Self-Healing" mechanism, which automatically adjusts convergence parameters (e.g., mixing beta, smearing) when a calculation fails. Additionally, we implement the **Periodic Embedding** algorithm, essential for extracting local environments from large simulations for efficient DFT calculation.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/
│       │   ├── **oracle.py**       # Core Oracle Logic
│       │   └── **base.py**         # (Updated ABCs)
│       ├── interfaces/
│       │   ├── **qe_driver.py**    # Quantum Espresso Wrapper
│       │   └── **pseudos.py**      # Pseudopotential Logic
│       └── utils/
│           └── **embedding.py**    # Periodic Embedding Logic
└── tests/
    ├── **test_oracle.py**
    └── **test_embedding.py**
```

## 3. Design Architecture

### 3.1 DFT Manager (`core/oracle.py`)
The `DFTManager` class orchestrates the calculation workflow. It accepts a list of `ase.Atoms` objects and returns their energies, forces, and stresses.
*   **Input**: `List[Atoms]`, `DFTConfig`.
*   **Output**: `List[Atoms]` (with `calc.results` attached).
*   **Configuration (`DFTConfig`)**:
    *   `mixing_beta` (float, default 0.7): Initial mixing parameter for SCF.
    *   `smearing_type` (str, default "mv"): Type of smearing (e.g., "mv", "gaussian").
    *   `smearing_width` (float, default 0.1): Width of smearing in eV.
    *   `diagonalization` (str, default "david"): Diagonalization algorithm.
    *   `pseudopotentials` (Dict[str, str]): Mapping of element symbols to filenames.
*   **Self-Healing Strategy**:
    1.  Attempt standard calculation.
    2.  `JobFailedException` caught.
    3.  Reduce `mixing_beta`. Retry.
    4.  Increase `smearing_width`. Retry.
    5.  Change `diagonalization`. Retry.
    6.  Fail gracefully if all attempts exhausted.

### 3.2 Periodic Embedding (`utils/embedding.py`)
This utility is crucial for the "Active Learning" loop.
*   **Function**: `embed_cluster(cluster: Atoms, buffer: float) -> Atoms`
*   **Logic**:
    1.  Create a bounding box around the cluster.
    2.  Add a vacuum buffer (e.g., 10 Å).
    3.  Create an orthorhombic cell.
    4.  Verify minimum image convention is satisfied.

### 3.3 Quantum Espresso Interface (`interfaces/qe_driver.py`)
Wraps `ase.calculators.espresso.Espresso`.
*   **Features**:
    *   Automatic k-point generation based on cell size (k-spacing).
    *   Pseudopotential dictionary management (SSSP) via `interfaces/pseudos.py`.
    *   Parallel execution management (`mpirun`).

## 4. Implementation Approach

### Step 1: Embedding Logic
*   Implement `src/pyacemaker/utils/embedding.py`.
*   Ensure it handles both cluster-in-vacuum and surface-slab geometries correctly.

### Step 2: QE Interface
*   Implement `src/pyacemaker/interfaces/qe_driver.py`.
*   Connect it to the `DFTManager`.

### Step 3: Self-Healing Logic
*   Implement the retry loop in `DFTManager`.
*   Define `HealableError` exceptions.

## 5. Test Strategy

### 5.1 Unit Testing (`test_embedding.py`)
*   **Geometry Check**: Create a simple dimer. Embed it. Verify the cell vectors are orthogonal and sufficiently large.
*   **PBC Check**: Verify that `pbc=[True, True, True]` is set on the output.

### 5.2 Integration Testing (`test_oracle.py`)
*   **Mock Execution**: Use a mock `ase.calculator` that simulates a convergence failure on the first attempt and success on the second. Verify that `DFTManager` retries and succeeds.
*   **Command Generation**: Verify that the generated `pw.x` input file contains the correct `mixing_beta` and `smearing` values.

### 5.3 Coverage Goals
*   100% coverage on embedding logic (corner cases like single atom).
*   Verify exception handling paths in `DFTManager`.
