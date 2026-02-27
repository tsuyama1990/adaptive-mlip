# Cycle 06: Integration, CLI, and UAT

## 1. Summary
Cycle 06 is the final integration phase. It focuses on wiring all the modules developed in Cycles 01-05 into a coherent, executable pipeline. We will polish the Command Line Interface (CLI) in `main.py`, finalize the `Orchestrator` to handle the full 7-step workflow seamlessly, and ensure that the system is robust against failures (e.g., resuming from a specific step).

Most importantly, this cycle involves the creation of the comprehensive `UAT_AND_TUTORIAL.py` notebook, which serves as both the final acceptance test and the primary user documentation.

## 2. System Architecture

### 2.1. File Structure
**Files to be created/modified in this cycle are bolded.**

```text
src/
└── pyacemaker/
    ├── **main.py**                     # Final CLI with arguments
    ├── **orchestrator.py**             # Full 7-step state machine
    └── utils/
        └── **logging.py**              # Enhanced logging (File + Console)
tutorials/
    └── **UAT_AND_TUTORIAL.py**         # The Master Tutorial
```

### 2.2. CLI Arguments (`main.py`)
*   `run`: Execute the pipeline.
    *   `--config`: Path to `config.yaml`.
    *   `--step`: Run only specific steps (e.g., `1,2` or `all`).
    *   `--resume`: Resume from the last successful step.
*   `init`: Create a template workspace.
    *   `--dir`: Directory to initialize.

### 2.3. Orchestrator Logic
The `Orchestrator` will manage a `WorkflowState` object, saved as `state.json`.
*   **State Tracking:** `{'step1_done': True, 'step2_done': False, ...}`.
*   **Artifact Management:** Ensures outputs from previous steps are available before starting the next.

## 3. Design Architecture

### 3.1. Orchestrator State Machine
*   **Load Config**
*   **Step 1:** Call `DirectSampler`. Save `data/step1.xyz`. Update State.
*   **Step 2:** Call `MaceOracle` (Uncertainty). Filter. Call `DftOracle`. Save `data/step2.xyz`. Update State.
*   **Step 3:** Call `MaceTrainer`. Save `models/mace_ft.model`. Update State.
*   **Step 4:** Call `MdGenerator`. Save `data/step4.xyz`. Update State.
*   **Step 5:** Call `MaceOracle` (Label). Save `data/step5.xyz`. Update State.
*   **Step 6:** Call `PacemakerTrainer`. Save `models/base.yace`. Update State.
*   **Step 7:** Call `PacemakerDeltaTrainer`. Save `models/final.yace`. Update State.

## 4. Implementation Approach

1.  **CLI Polish:** Use `argparse` or `click` (if added to deps) to build a robust CLI.
2.  **Orchestrator:** Implement the full sequential logic with error handling (try/except blocks).
3.  **Tutorial:** Write `tutorials/UAT_AND_TUTORIAL.py` using `marimo`. This file will import the `Orchestrator` and run it in a notebook environment, visualizing results at each step.

## 5. Test Strategy

### 5.1. End-to-End Test
*   **Mock Mode:** Run `pyacemaker run --config examples/mock.yaml`.
*   **Verify:** All 7 steps complete, and `final.yace` exists.

### 5.2. Resume Functionality
*   **Interrupt:** Run steps 1-3.
*   **Resume:** Run with `--resume` and verify it starts at Step 4.
