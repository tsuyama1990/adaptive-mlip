# Cycle 02 UAT: The Oracle (DFT Automation)

## 1. Test Scenarios

### Scenario 02-01: "Single Point Calculation" (Priority: High)
**Objective**: Verify that the system can run a simple DFT calculation (or mock equivalent).
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 2 - Oracle)

1.  **Preparation**:
    *   Create a file `h2o.xyz` containing a water molecule.
    *   Set `config.yaml` to use a "Mock Calculator" (returns LJ potential).
2.  **Action**: Run a script that instantiates `DFTManager` and calls `compute([h2o_atoms])`.
3.  **Expectation**:
    *   The returned atoms object has `.get_potential_energy()` populated.
    *   The returned value is consistent (e.g., -14.5 eV for the mock).
    *   Forces are populated (3xN array).

### Scenario 02-02: "Self-Healing Test" (Priority: Medium)
**Objective**: Verify that the system recovers from a simulated SCF convergence failure.

1.  **Preparation**:
    *   Configure the Mock Calculator to raise `SCFError` on the first call, then succeed.
2.  **Action**: Run `DFTManager.compute()`.
3.  **Expectation**:
    *   Log output shows: "SCF failed. Retrying with mixing_beta=0.3...".
    *   Final result is successful.
    *   Process does not crash.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Automation

  Scenario: Calculate Energy of a Structure
    GIVEN an atomic structure "H2O"
    AND a valid DFT configuration (pseudopotentials, k-points)
    WHEN the Oracle computes the energy
    THEN the system should generate a valid input file for Quantum Espresso
    AND execute the calculation
    AND parse the output to retrieve Energy, Forces, and Stress

  Scenario: Recover from Convergence Failure
    GIVEN a difficult electronic structure
    WHEN the Oracle encounters an SCF convergence error
    THEN the system should catch the error
    AND the system should modify the mixing parameters (beta)
    AND retry the calculation automatically
    AND return the converged result without user intervention
```
