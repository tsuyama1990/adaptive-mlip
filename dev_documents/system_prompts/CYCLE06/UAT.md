# Cycle 06 UAT: Full Orchestration & SN2 Polish

## 1. Test Scenarios

### Scenario 6.1: Full Pipeline (Mock Mode)
*   **ID**: UAT-06-01
*   **Priority**: Critical
*   **Description**: Run the complete 7-step workflow end-to-end using mock data/models.
*   **Pre-conditions**:
    *   `config.yaml` exists.
    *   Environment variable `PYACEMAKER_MODE=MOCK` is set.
*   **Post-conditions**:
    *   All 7 steps complete successfully.
    *   Final artifacts (`final_potential.yace`) exist.
    *   Exit code 0.

### Scenario 6.2: Workflow Resume
*   **ID**: UAT-06-02
*   **Priority**: High
*   **Description**: Verify that the pipeline can resume from a failure point without re-running expensive steps.
*   **Pre-conditions**:
    *   Workflow failed at Step 4.
    *   `workflow_state.json` indicates Step 3 complete.
*   **Post-conditions**:
    *   Running the command again starts directly at Step 4.
    *   Step 1-3 artifacts are preserved (timestamps unchanged).
    *   Workflow completes Step 4-7.

### Scenario 6.3: SN2 Tutorial Execution
*   **ID**: UAT-06-03
*   **Priority**: Medium
*   **Description**: Execute the `UAT_AND_TUTORIAL.py` Marimo notebook to verify the user experience.
*   **Pre-conditions**:
    *   `tutorials/UAT_AND_TUTORIAL.py` exists.
    *   Dependencies installed.
*   **Post-conditions**:
    *   The script runs without error.
    *   Plots are generated (saved as PNG or displayed if interactive).
    *   The "SN2 Barrier" calculated by the final potential matches the reference (within tolerance).

## 2. Behavior Definitions (Gherkin)

### Feature: End-to-End Execution

```gherkin
Feature: Full Workflow
  As a user
  I want to run the entire pipeline with a single command
  So that I don't have to manually manage each step

  Scenario: Running SN2 Reaction
    Given a valid configuration "sn2_config.yaml"
    When I run the command "pyacemaker run --all"
    Then the process should complete successfully
    And the final output "final_potential.yace" should be valid
    And the log should show "Step 1... Done", "Step 2... Done", etc.
```

### Feature: Resume Capability

```gherkin
Feature: Fault Tolerance
  As a user
  I want to resume my long-running job if the cluster kills it
  So that I don't lose days of computation

  Scenario: Resuming from Step 4
    Given a workflow state file indicating Step 3 is DONE
    When I run the command "pyacemaker run --resume"
    Then the system should skip Steps 1, 2, and 3
    And the system should start execution at Step 4
    And the log should say "Resuming from Step 4"
```
