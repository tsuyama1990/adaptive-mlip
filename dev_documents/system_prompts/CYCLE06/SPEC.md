# Cycle 06 Specification: Full Orchestration & SN2 Polish

## 1. Summary
This final cycle focuses on **Production Readiness** and **User Acceptance**. Now that all individual components (Steps 1-7) are implemented, we must ensure they work seamlessly together as a robust, resilient pipeline.

We will finalize the `Orchestrator` to support full state persistence, allowing the workflow to be paused and resumed (a critical feature for long-running scientific jobs). We will also implement the **SN2 Reaction Tutorial** as a single executable Marimo notebook (`UAT_AND_TUTORIAL.py`), which serves as both the primary user documentation and the definitive integration test for the system. Finally, we will polish the Command Line Interface (CLI) to provide a smooth user experience.

## 2. System Architecture

The following file structure will be created or modified. Files marked in **bold** are the focus of this cycle.

```text
pyacemaker/
├── **tutorials/**
│   └── **UAT_AND_TUTORIAL.py** # The Marimo Notebook
├── src/
│   └── pyacemaker/
│       ├── **orchestrator.py**     # State persistence & Resume logic
│       ├── **main.py**             # Final CLI polish
│       ├── **constants.py**        # Error codes & status messages
│       └── utils/
│           └── **state_manager.py** # Helper for atomic state writes
└── tests/
    └── system/
        └── **test_end_to_end.py**
```

## 3. Design Architecture

### 3.1. State Persistence (`utils/state_manager.py`)

*   **`StateManager`**:
    *   **Responsibility**: Atomically write `workflow_state.json` to disk.
    *   **Logic**: Use a temporary file + rename strategy to prevent corruption if the process is killed mid-write.
    *   **Schema**:
        ```json
        {
          "current_step": 4,
          "status": "RUNNING",
          "artifacts": {
            "step1_xyz": "path/to/file",
            "mace_model": "path/to/model"
          },
          "errors": []
        }
        ```

### 3.2. Orchestrator Logic (`orchestrator.py`)

*   **`resume_workflow()`**:
    *   Reads `workflow_state.json`.
    *   Determines the last successfully completed step.
    *   Skips re-execution of completed steps.
    *   Example: If Step 3 failed, it re-loads Step 2 artifacts and restarts Step 3.

### 3.3. The Tutorial (`tutorials/UAT_AND_TUTORIAL.py`)

*   **Format**: A Marimo notebook (Python script with metadata).
*   **Content**:
    1.  **Setup**: Install deps (if needed), set `PYACEMAKER_MODE`.
    2.  **Config**: Define the SN2 Reaction parameters inline.
    3.  **Execution**: Call `Orchestrator` API directly to run steps.
    4.  **Visualization**: Use `matplotlib` or `ase.visualize` (if available) to show the "Uncertainty vs. Energy" plot and the final "NEB Barrier" plot.
    5.  **Validation**: Assert that the final barrier height is within 0.05 eV of the reference (in Real Mode) or that the file exists (in Mock Mode).

## 4. Implementation Approach

### Step 4.1: State Management
1.  Implement `StateManager`.
2.  Update `Orchestrator` to call `save_state()` after every step.
3.  Implement `load_state()` at startup.

### Step 4.2: CLI Polish
1.  Use `argparse` or `click` (if added) to support:
    *   `pyacemaker init`: Create config.
    *   `pyacemaker run`: Run full pipeline.
    *   `pyacemaker run --step N`: Run specific step.
    *   `pyacemaker clean`: Remove work directory.

### Step 4.3: Tutorial Implementation
1.  Write the `UAT_AND_TUTORIAL.py`.
2.  Implement the SN2 specific logic:
    *   Reaction coordinate definition (C-Cl distance).
    *   Mock Oracle logic for SN2 (a simple 1D potential surface for the C-Cl bond).

## 5. Test Strategy

### 5.1. System Testing
*   **`test_end_to_end.py`**:
    *   **Goal**: Run the entire pipeline from `init` to `final`.
    *   **Mode**: Mock Mode (CI).
    *   **Assertions**:
        *   Process exit code is 0.
        *   Final `.yace` file exists.
        *   Log file contains "Workflow Completed Successfully".

### 5.2. Resume Testing
*   **`test_resume.py`**:
    *   Start pipeline, force kill at Step 3 (simulate crash).
    *   Restart pipeline.
    *   Assert that Step 1 and 2 are NOT re-executed (check timestamps of log/files).
    *   Assert that Step 3 runs and completes.

### 5.3. Tutorial Verification
*   **Manual/Automated**: Run `marimo run tutorials/UAT_AND_TUTORIAL.py`. Check for no exceptions.
