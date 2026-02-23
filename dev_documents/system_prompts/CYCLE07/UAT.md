# Cycle 07 UAT: The Guardian (Validation & QA)

## 1. Test Scenarios

### Scenario 07-01: "Validate Potential" (Priority: High)
**Objective**: Verify that the system can run a full validation suite on a potential.
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 6 - Validation)

1.  **Preparation**:
    *   Use the potential from Cycle 06.
    *   Set `config.yaml` to enable Phonon and Elastic checks.
2.  **Action**: Run `Validator.validate()`.
3.  **Expectation**:
    *   Process completes.
    *   An HTML report `validation_report.html` is created.
    *   Console output shows "Validation PASSED" (or FAILED with reason).

### Scenario 07-02: "Unstable Detection" (Priority: Medium)
**Objective**: Verify that the system flags an unstable potential.

1.  **Preparation**:
    *   Use a dummy potential known to be bad (or artificially tweak parameters).
2.  **Action**: Run `Validator.validate()`.
3.  **Expectation**:
    *   Console output shows "Validation FAILED".
    *   Report highlights "Imaginary Frequencies Detected".

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Validation

  Scenario: Verify Dynamical Stability
    GIVEN a candidate potential
    WHEN the Validator runs phonon calculations
    THEN it should compute the phonon band structure
    AND verify the absence of imaginary frequencies (instability)
    AND report PASS/FAIL

  Scenario: Verify Mechanical Stability
    GIVEN a candidate potential
    WHEN the Validator runs elastic constant calculations
    THEN it should compute the stiffness tensor (Cij)
    AND verify Born stability criteria for the crystal system
    AND report PASS/FAIL
```
