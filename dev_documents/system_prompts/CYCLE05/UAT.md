# Cycle 05 UAT: The Engine (MD & Inference)

## 1. Test Scenarios

### Scenario 05-01: "Run Hybrid MD" (Priority: High)
**Objective**: Verify that the system can run an MD simulation using a hybrid (ACE+ZBL) potential.
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 4 - Engine)

1.  **Preparation**:
    *   Use the potential generated in Cycle 04.
    *   Set `config.yaml` to run NPT at 300K for 1000 steps.
2.  **Action**: Run `MDEngine.run()`.
3.  **Expectation**:
    *   Simulation completes successfully.
    *   Trajectory file `dump.lammpstrj` is created.
    *   Log file confirms `pair_style hybrid/overlay` was used.

### Scenario 05-02: "Uncertainty Halt" (Priority: Medium)
**Objective**: Verify that the simulation stops when uncertainty is high.

1.  **Preparation**:
    *   Set `config.yaml` with a very low uncertainty threshold (e.g., `gamma_threshold: 0.01`) to force a halt.
2.  **Action**: Run `MDEngine.run()`.
3.  **Expectation**:
    *   Simulation stops before completing all steps.
    *   The engine returns a status indicating `HALTED`.
    *   The snapshot causing the halt is extracted.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Molecular Dynamics Engine

  Scenario: Execute Hybrid MD
    GIVEN a trained ACE potential and a ZBL baseline
    WHEN the Engine starts an MD simulation
    THEN it should configure LAMMPS to use a hybrid/overlay pair style
    AND ensuring physical robustness against atomic collisions

  Scenario: Halt on Uncertainty
    GIVEN a running simulation
    WHEN the extrapolation grade (gamma) of any atom exceeds the safety threshold
    THEN the Engine should immediately interrupt the simulation
    AND save the current atomic configuration for further analysis
    AND report the "Halted" status to the Orchestrator
```
