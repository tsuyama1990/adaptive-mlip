# Cycle 04 Specification: The Trainer (Pacemaker Integration)

## 1. Summary
This cycle implements the **Trainer** module, which wraps the **Pacemaker** library to fit ACE potentials. A critical component is the **Delta Learning** strategy, where we fit only the difference between the DFT energy and a physics-based baseline (LJ/ZBL). This ensures robustness in the high-energy repulsive regime. We also integrate **Active Set Optimization** (D-optimality) to select the most informative structures for training, significantly reducing computational cost.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/
│       │   ├── **trainer.py**      # Pacemaker Wrapper
│       │   └── **active_set.py**   # D-Optimality Logic
│       └── utils/
│           └── **delta.py**        # Baseline (LJ/ZBL) Logic
└── tests/
    ├── **test_trainer.py**
    └── **test_delta.py**
```

## 3. Design Architecture

### 3.1 Pacemaker Wrapper (`core/trainer.py`)
Manages the `pace_train` execution.
*   **Input**: `List[Atoms]` (labelled), `TrainingConfig` (including `active_set_size`).
*   **Output**: `potential.yace` file path.
*   **Features**:
    *   Generates `input.yaml` for Pacemaker.
    *   Handles `subprocess` execution of `pace_train`.
    *   Parses training logs for RMSE.

### 3.2 Delta Learning (`utils/delta.py`)
Calculates baseline parameters.
*   **Function**: `get_lj_params(elements) -> Dict`
*   **Logic**: Uses standard atomic radii/epsilon tables or ZBL formulas.

### 3.3 Active Set Selection (`core/active_set.py`)
Wraps `pace_activeset`.
*   **Function**: `select_active_set(candidates: List[Atoms], current_potential, n_select: int) -> List[Atoms]`
*   **Logic**: Maximizes the determinant of the descriptor matrix. The number of structures to select is controlled by `active_set_size` in `TrainingConfig`.

## 4. Implementation Approach

### Step 1: Delta Learning Utils
*   Implement `src/pyacemaker/utils/delta.py`.
*   Ensure it covers the periodic table (or a large subset).

### Step 2: Active Set Logic
*   Implement `src/pyacemaker/core/active_set.py`.
*   Interface with Pacemaker's CLI or API.

### Step 3: Trainer Orchestration
*   Implement `src/pyacemaker/core/trainer.py`.
*   Construct the command line for `pace_train`.
*   Implement the `fit()` method.

## 5. Test Strategy

### 5.1 Unit Testing (`test_delta.py`)
*   **Parameter Check**: Verify that `get_lj_params("Fe")` returns reasonable sigma/epsilon values.
*   **ZBL Check**: Verify ZBL potential form for very short distances.

### 5.2 Integration Testing (`test_trainer.py`)
*   **End-to-End**: Mock `pace_train` (create a dummy file) and verify that `Trainer.fit()` returns the expected path.
*   **Config Gen**: Verify that the generated `input.yaml` contains the correct `cutoff` and `elements`.

### 5.3 Coverage Goals
*   100% coverage on `delta.py`.
*   Verify command construction in `trainer.py`.
