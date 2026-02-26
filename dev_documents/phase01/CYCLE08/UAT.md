# Cycle 08 UAT: The Expander (kMC & Production)

## 1. Test Scenarios

### Scenario 08-01: "Grand Challenge" (Priority: Critical)
**Objective**: Verify the full Fe/Pt on MgO deposition and ordering workflow.
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 7 - Grand Challenge)

1.  **Preparation**:
    *   Set `config.yaml` to run the `fept_mgo` scenario.
    *   Mode: Mock (CI) or Real (Local).
2.  **Action**: Run `python -m pyacemaker.main --scenario fept_mgo`.
3.  **Expectation**:
    *   Step 1 (Surface Gen): `mgo_slab.xyz` created.
    *   Step 2 (Deposition): `deposited.xyz` created (shows adatoms).
    *   Step 3 (Ordering): `ordered.xyz` created (shows L10 pattern).
    *   Log output confirms success.

### Scenario 08-02: "EON kMC Run" (Priority: High)
**Objective**: Verify that the EON interface works correctly.

1.  **Preparation**:
    *   Use a simple adatom diffusion system (e.g., Pt on Pt).
2.  **Action**: Run `EONWrapper.run()`.
3.  **Expectation**:
    *   EON explores transition states.
    *   A table of barriers is produced.
    *   The system evolves over time (KMC steps > 0).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Advanced Simulation Scenarios

  Scenario: Simulate Long-Term Ordering
    GIVEN a disordered alloy surface created by MD deposition
    WHEN the system switches to Adaptive Kinetic Monte Carlo (aKMC)
    THEN it should explore rare events (diffusion, exchange)
    AND evolve the system towards the thermodynamic equilibrium (L10 ordered phase)
    AND reach time scales inaccessible to standard MD
```
