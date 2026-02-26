# Cycle 03 UAT: The Explorer (Structure Generation)

## 1. Test Scenarios

### Scenario 03-01: "Generate Candidates" (Priority: High)
**Objective**: Verify that the system can generate a set of perturbed structures from a base composition.
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 2 - Explorer)

1.  **Preparation**:
    *   Set `config.yaml` with `composition: FePt` and `policy: random_rattle`.
2.  **Action**: Run `StructureGenerator.generate(n=10)`.
3.  **Expectation**:
    *   The system returns a list of 10 `ase.Atoms` objects.
    *   Each structure is slightly different (verify positions).
    *   All structures have the correct chemical formula (FePt).

### Scenario 03-02: "Defect Generation" (Priority: Medium)
**Objective**: Verify that the system can introduce vacancies.

1.  **Preparation**:
    *   Set `config.yaml` with `policy: defects` and `vacancy_rate: 0.05`.
2.  **Action**: Run `StructureGenerator.generate(n=1)`.
3.  **Expectation**:
    *   The returned structure has fewer atoms than the pristine bulk.
    *   The lattice vectors remain unchanged (or relax slightly if mock relaxation is enabled).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Structure Exploration

  Scenario: Generate Initial Guess
    GIVEN a chemical composition "FePt"
    WHEN the Generator executes the "Cold Start" policy
    THEN the system should query M3GNet (or database)
    AND return a relaxed crystal structure of FePt (L10 phase)

  Scenario: Perturb Structure
    GIVEN a base structure
    WHEN the Generator applies "Rattle" perturbation (sigma=0.1)
    THEN the atomic positions should be displaced by random Gaussian noise
    AND the total number of atoms should remain constant
```
