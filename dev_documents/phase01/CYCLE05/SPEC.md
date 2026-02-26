# Cycle 05 Specification: The Engine (MD & Inference)

## 1. Summary
This cycle implements the **Dynamics Engine**, the module responsible for running Molecular Dynamics (MD) simulations using **LAMMPS**. Key features include the automatic generation of **Hybrid Potentials** (overlaying ACE with a physical baseline like ZBL/LJ) to ensure safety during high-energy events, and the **Uncertainty Watchdog**, which uses `fix halt` to stop the simulation immediately when the extrapolation grade ($\gamma$) exceeds a safe threshold. This transforms MD from a passive simulation tool into an active data-gathering instrument.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/
│       │   ├── **engine.py**       # Core Engine Logic
│       │   └── **base.py**         # (Updated ABCs)
│       └── interfaces/
│           └── **lammps_driver.py** # LAMMPS Python Interface
└── tests/
    └── **test_engine.py**
```

## 3. Design Architecture

### 3.1 MD Interface (`core/engine.py`)
Orchestrates the MD workflow.
*   **Input**: `Atoms` (initial structure), `MDConfig`, `PotentialPath`.
*   **Output**: `TrajectoryFile` (dump), `HaltInfo` (if stopped).
*   **Logic**:
    *   Constructs the LAMMPS input script dynamically.
    *   Injects `pair_style hybrid/overlay`.
    *   Configures `compute pace` and `fix halt`.

### 3.2 LAMMPS Driver (`interfaces/lammps_driver.py`)
Wraps the `lammps` Python module.
*   **Features**:
    *   `run(script: str)`: Executes a raw LAMMPS script.
    *   `extract_variable(name: str)`: Retrieves values from LAMMPS during/after run.
    *   `get_atoms()`: Converts current LAMMPS state back to ASE Atoms.

## 4. Implementation Approach

### Step 1: LAMMPS Wrapper
*   Implement `src/pyacemaker/interfaces/lammps_driver.py`.
*   Ensure it can load the `pace` pair style (requires `USER-PACE` package).

### Step 2: Hybrid Potential Logic
*   Implement the logic to generate `pair_coeff` commands that overlay ACE with ZBL/LJ.
*   Ensure the parameters match those used in Training (Cycle 04).

### Step 3: Uncertainty Watchdog
*   Implement the `fix halt` logic in `MDInterface`.
*   Define the parsing logic to identify *which* atom caused the halt.

## 5. Test Strategy

### 5.1 Unit Testing (`test_engine.py`)
*   **Script Generation**: Verify that the generated LAMMPS script contains the correct `pair_style hybrid/overlay` command and coefficients.
*   **Halt Logic**: Verify that the script includes `fix halt` with the user-defined threshold.

### 5.2 Integration Testing (`test_engine.py`)
*   **Run MD**: Execute a short MD run (100 steps) on a test system (e.g., Al) using a dummy potential. Verify that a trajectory file is created.
*   **Trigger Halt**: (Advanced) Mock the `compute pace` to artificially return a high gamma value and verify that the simulation stops early.

### 5.3 Coverage Goals
*   100% coverage on input script generation logic.
*   Verify robust error handling if LAMMPS crashes.
