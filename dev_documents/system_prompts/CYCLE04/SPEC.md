# CYCLE 04: Orchestration & HPC Resilience

## 1. Summary

This final cycle brings together the previous three cycles by fundamentally overhauling the central `Orchestrator` (`src/pyacemaker/core/loop.py` and `orchestrator.py`). The objective is to implement the full 4-Phase "Hierarchical Distillation Architecture" state machine, replacing the simple iterative active learning loop. Furthermore, this cycle addresses the non-functional requirements for HPC deployment: robust, task-level checkpointing (to survive node failures) and aggressive artifact cleanup (to prevent storage quotas from being exceeded during million-step MD runs).

## 2. System Architecture

The scope involves the highest level of control flow and state management.

```text
src/pyacemaker/
├── core/
│   ├── base.py                 (No changes)
│   ├── engine.py               (No changes)
│   ├── oracle.py               (No changes)
│   ├── trainer.py              (No changes)
│   ├── **loop.py**             (Modify: Implement 4-Phase state machine)
│   └── **state_manager.py**    (Modify: Fine-grained SQLite/JSON checkpoints)
├── domain_models/
│   ├── config.py               (No changes)
│   └── **data.py**             (Modify: State models for checkpoints)
└── utils/
    └── **cleanup.py**          (Create: Artifact management daemon/functions)
```

## 3. Design Architecture

The `Orchestrator` must transition from a simple `while True` loop to a robust State Machine pattern, capable of persisting and resuming its exact state at any point.

### 3.1. Orchestrator & State Machine (`core/loop.py` & `orchestrator.py`)

The `Orchestrator` will manage the execution of the four phases:

*   **Phase 1: Zero-Shot Distillation:** Generate combinatorial structures -> DIRECT sampling -> MACE filtering -> Baseline ACE training. (Executes only once per project if `DistillationConfig.enable` is True).
*   **Phase 2: Validation:** Run EOS, Elastic, Phonon, and Miniature MD stress tests. (Executes after Phase 1, or after major Phase 4 updates).
*   **Phase 3 & 4 (The Production Loop):** This is the "Inverted" loop.
    1.  Start/Resume the main LAMMPS MD engine (Cycle 02).
    2.  Engine runs until the Uncertainty Watchdog triggers a `HALT`.
    3.  `Orchestrator` catches the `HALT`, receives the step number and "epicenter" atom indices.
    4.  Execute `extract_intelligent_cluster` (Cycle 01).
    5.  Pass cluster to `QEDriver` (DFT).
    6.  Execute `FinetuneManager` (MACE) and Surrogate Generation (Cycle 03).
    7.  Execute `IncrementalTrainer` (Cycle 03).
    8.  Update state, increment generation counter, and loop back to step 1 (Seamless Resume).

### 3.2. State Manager & Checkpointing (`core/state_manager.py`)

*   **Task-Level Granularity:** Instead of saving state only at the end of a full iteration, the `StateManager` must commit to disk after *every major sub-task* (e.g., after Phase 1 completes, after a single DFT calculation finishes, after a surrogate batch is generated).
*   **Implementation:** Use a robust JSON file (`state.json`) with atomic writes (write to temp file, then rename) or a lightweight SQLite database to prevent corruption if the job is killed mid-write.
*   **Data Model:** The checkpoint must store the current Phase, the current MD step (if in Phase 3/4), paths to the current best potential (`.yace`), and the status of the Replay Buffer.

### 3.3. Artifact Cleanup (`utils/cleanup.py`)

*   **Large File Deletion:** Create utility functions to systematically find and delete or gzip Quantum Espresso `.wfc` (wavefunction) files and massive LAMMPS `.dump` files immediately after they have been successfully processed and their relevant data extracted into ASE `Atoms` objects.
*   **Daemon/Async (Optional but recommended):** If possible without adding heavy dependencies, run cleanup tasks asynchronously so they don't block the critical path of the active learning loop.

## 4. Implementation Approach

1.  **Refactor `Orchestrator.run()`:**
    *   Change the core logic to a `while True:` loop governed by a `current_phase` state variable loaded from the `StateManager`.
    *   Implement `_run_phase1()`, `_run_phase2()`, and `_run_production_loop()`.
    *   In `_run_production_loop()`, implement the `try-except` block to catch the `MDHaltError` from the LAMMPS engine, triggering the Phase 3 (Extraction/DFT) and Phase 4 (Training) sequence.
2.  **Upgrade `StateManager`:**
    *   Modify the `save_state()` method to use atomic file operations: `with open('state.json.tmp', 'w') as f: json.dump(...)` followed by `os.replace('state.json.tmp', 'state.json')`.
    *   Add fine-grained logging and status fields to the `LoopState` Pydantic model (e.g., `last_completed_task: str`).
3.  **Implement Cleanup Utilities:**
    *   Create `utils/cleanup.py` with a function `cleanup_artifacts(directory: Path, patterns: List[str] = ['*.wfc', '*.save', 'dump.*'])`.
    *   Inject this function call at the end of the `QEDriver` execution block and the LAMMPS engine evaluation block in the Orchestrator.
4.  **Refine & Lint:** Run Ruff and MyPy. Pay close attention to exception handling (`tryceratops` rules) to ensure no exceptions are silently swallowed during the complex state machine transitions.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)

Unit testing will rigorously test the State Machine logic and the atomicity of the checkpointing system.

1.  **State Machine Transitions (`Orchestrator`):**
    *   Mock all underlying engines (`LAMMPSEngine`, `QEDriver`, `IncrementalTrainer`, `extract_intelligent_cluster`).
    *   Initialize the Orchestrator with a clean state.
    *   **Assertions:**
        *   If `DistillationConfig.enable=True`, the Orchestrator must call `_run_phase1()`, update the state to `PHASE_2`, and save.
        *   Mock the LAMMPS engine to raise an `MDHaltError(step=5000, epicenter=[10, 11])`.
        *   The Orchestrator must catch it, call the extraction function, pass the result to DFT, call the Trainer, update the state, and loop back to the engine with `restart_file` pointing to step 5000.
2.  **Atomic Checkpointing (`StateManager`):**
    *   Instantiate a `StateManager`.
    *   Call `save_state()`. Verify `state.json` is created correctly.
    *   **Simulate a Crash:** Mock the `os.replace` function to raise a generic `OSError` *after* the temporary file is written.
    *   Call `save_state()` again.
    *   **Assertions:** The original `state.json` must remain uncorrupted and parseable, preventing the entire workflow from becoming unresumable due to an I/O error during save.
3.  **Artifact Cleanup Logic:**
    *   Create a temporary directory with dummy `.wfc`, `.save` directories, and a `dump.lammps` file, alongside legitimate `.yaml` and `.xyz` files.
    *   Call `cleanup_artifacts()`.
    *   **Assertions:** Only the specified large files/directories are deleted or gzipped. The configuration and structure files must remain untouched.

### Integration Testing Approach (Min 300 words)

The final integration test will simulate a complete, multi-phase execution of the PyAceMaker pipeline using fast mock models, validating HPC resilience.

1.  **The "Kill and Resume" E2E Test:**
    *   Set up a complete configuration YAML mimicking a production run, but using fast Mocks for all heavy calculations (Mock MACE, Mock DFT, Mock Pacemaker).
    *   Launch the `Orchestrator` via a subprocess (`subprocess.Popen`).
    *   Allow the process to run through Phase 1 and enter the Production Loop (Phase 3/4).
    *   Send a `SIGKILL` (`kill -9`) to the Python subprocess *while* it is executing a mocked "DFT calculation" (Phase 3). This simulates a Slurm Wall-time timeout.
    *   **Assertions:** Inspect the file system. A `state.json` must exist, recording that Phase 1 is complete and the system was in the middle of Phase 3.
    *   Relaunch the `Orchestrator` subprocess in the same directory.
    *   **Assertions:** The system must parse `state.json`, recognize Phase 1 is done, skip it entirely, and immediately attempt to re-run the interrupted DFT calculation from Phase 3, successfully completing the loop and returning to Phase 4.
2.  **Storage Footprint Verification:**
    *   Run a mock pipeline that performs 10 cycles of the Production Loop (10 DFT calls, 10 LAMMPS runs).
    *   **Assertions:** The total disk space used by the working directory must remain roughly constant (O(1)) after the first few cycles, proving that the `cleanup_artifacts` utility successfully prevents unbounded storage growth from `.wfc` and `dump` files.
