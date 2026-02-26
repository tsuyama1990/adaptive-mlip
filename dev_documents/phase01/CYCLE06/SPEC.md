# Cycle 06 Specification: The Orchestrator (Active Learning Loop)

## 1. Summary
This cycle implements the core logic of the **Active Learning Loop**, connecting all the individual components (Generator, Oracle, Trainer, Engine) into a unified, self-driving system. The **Orchestrator** manages the entire lifecycle: iterating through exploration, detection, selection, calculation, and refinement. It handles state transitions (e.g., "Exploration" -> "Halt" -> "Refinement") and ensures data persistence, allowing the system to resume from interruptions (e.g., power failure) without data loss.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── **orchestrator.py**     # (Updated Loop Logic)
│       └── core/
│           └── **loop.py**         # (State Machine Logic)
└── tests/
    └── **test_loop.py**
```

## 3. Design Architecture

### 3.1 Loop State Machine (`core/loop.py`)
Encapsulates the current state of the active learning campaign.
*   **Attributes**: `iteration: int`, `status: Enum("RUNNING", "HALTED", "CONVERGED")`, `current_potential: Path`.
*   **Methods**: `save_state()`, `load_state()`.

### 3.2 Orchestrator Updates (`orchestrator.py`)
Implements the main `run_loop()` method.
*   **Logic**:
    1.  Load Config & State.
    2.  Check for Initial Potential. If missing -> `Generator` -> `Oracle` -> `Trainer`.
    3.  Enter Loop:
        a.  Run MD (`Engine`).
        b.  If `HALTED`:
            i.   Extract Snapshot.
            ii.  `ActiveSet` selection (local).
            iii. `Oracle` calculation.
            iv.  `Trainer` update (fine-tuning).
        c.  If `CONVERGED` or `MAX_ITERS`: Break.
    4.  Finalize.

## 4. Implementation Approach

### Step 1: State Management
*   Implement `src/pyacemaker/core/loop.py`.
*   Use JSON/Pickle to save state to disk (`state.json`).

### Step 2: Loop Integration
*   Update `src/pyacemaker/orchestrator.py` to call `Generator`, `Oracle`, `Trainer`, `Engine` in sequence.
*   Implement the "Resume" logic (check for existing state on startup).

### Step 3: Deployment Logic
*   Implement logic to copy the best `potential.yace` to a `production/` folder.

## 5. Test Strategy

### 5.1 Unit Testing (`test_loop.py`)
*   **State Save/Load**: Create a dummy state object, save it, load it back, and verify equality.
*   **Iteration Limit**: Verify the loop stops exactly at `max_iterations`.

### 5.2 Integration Testing (`test_loop.py`)
*   **End-to-End Mock**: Run the full loop with all components mocked.
    *   Mock Generator returns 1 structure.
    *   Mock Oracle returns energy=-10.
    *   Mock Trainer returns "pot_v1.yace".
    *   Mock Engine runs 1 step and returns "HALTED" (simulated).
    *   Verify the Orchestrator proceeds to retrain and update to "pot_v2.yace".

### 5.3 Coverage Goals
*   100% coverage on loop control flow logic.
*   Verify robust exception handling (e.g., if Trainer fails, loop should pause/log error).
