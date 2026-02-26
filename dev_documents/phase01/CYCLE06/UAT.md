# Cycle 06 UAT: The Orchestrator (Active Learning Loop)

## 1. Test Scenarios

### Scenario 06-01: "Active Learning Campaign" (Priority: High)
**Objective**: Verify that the system can run a complete active learning loop from start to finish (mocked).
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 5 - Loop)

1.  **Preparation**:
    *   Set `config.yaml` with `max_iterations: 2` and `mode: mock`.
2.  **Action**: Run `Orchestrator.run()`.
3.  **Expectation**:
    *   The loop runs for exactly 2 iterations.
    *   Log output shows the sequence: "Exploration" -> "Halt" -> "Refinement".
    *   A `production/potential_v2.yace` file is created.

### Scenario 06-02: "Resume Capability" (Priority: Medium)
**Objective**: Verify that the system can resume a campaign after interruption.

1.  **Preparation**:
    *   Run the loop for 1 iteration, then interrupt (Ctrl+C simulation or forced exit).
    *   Verify `state.json` exists and shows `iteration: 1`.
2.  **Action**: Run `Orchestrator.run()` again.
3.  **Expectation**:
    *   The system loads the state.
    *   Log says "Resuming from iteration 1...".
    *   It continues to iteration 2 and finishes.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Orchestration

  Scenario: Autonomous Improvement
    GIVEN a starting potential
    WHEN the Orchestrator detects high uncertainty in MD
    THEN it should automatically trigger the refinement cycle
    AND update the potential with new DFT data
    AND resume the simulation with the improved potential

  Scenario: Persistent State
    GIVEN a long-running campaign
    WHEN the process is interrupted (e.g., power loss)
    THEN the system should have saved its progress
    AND upon restart, resume from the last completed checkpoint
```
