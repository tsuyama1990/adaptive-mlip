# Cycle 02 UAT: MACE Oracle & Active Learning

## 1. Test Scenarios

### Scenario 2.1: MACE Loading & Uncertainty Evaluation
*   **ID**: UAT-02-01
*   **Priority**: High
*   **Description**: The system must load a MACE model (or a valid mock/surrogate) and compute an uncertainty score for a given structure.
*   **Pre-conditions**: A valid structure (e.g., from `step1.xyz`).
*   **Post-conditions**:
    *   The model loads without error.
    *   The `uncertainty` field in the `AtomStructure` metadata is populated with a non-negative float.

### Scenario 2.2: Active Learning Selection Logic
*   **ID**: UAT-02-02
*   **Priority**: Critical
*   **Description**: Given a pool of structures, the system must correctly identify and select the top `N` most uncertain structures for DFT calculation.
*   **Pre-conditions**:
    *   `config.yaml` specifies `n_selection: 5`.
    *   `step1_initial.xyz` contains 10 structures.
    *   The mock MACE returns variable uncertainty scores.
*   **Post-conditions**:
    *   `step2_active_set.xyz` is created.
    *   It contains exactly 5 structures.
    *   These 5 structures correspond to the highest uncertainty scores from the pool.

### Scenario 2.3: Mock DFT Labeling
*   **ID**: UAT-02-03
*   **Priority**: Medium
*   **Description**: The system must simulate DFT labeling for the selected structures.
*   **Pre-conditions**: Step 2 selection is complete.
*   **Post-conditions**:
    *   The structures in `step2_active_set.xyz` have valid `energy` and `forces` arrays.
    *   The `provenance` metadata indicates "DFT_LABEL" (or "MOCK_DFT").

## 2. Behavior Definitions (Gherkin)

### Feature: MACE-based Filtering

```gherkin
Feature: Active Learning Filter
  As a researcher
  I want to select only the most informative structures
  So that I minimize expensive DFT calculations

  Scenario: Filtering Top 3 Uncertain Structures
    Given a pool of 5 structures with uncertainty scores:
      | id | uncertainty |
      | A  | 0.1         |
      | B  | 0.9         |
      | C  | 0.5         |
      | D  | 0.8         |
      | E  | 0.2         |
    And a configuration setting "n_selection" to 3
    When I run the active learning step
    Then the selected structures should be ["B", "D", "C"]
    And the output file "step2_active_set.xyz" should contain 3 frames
```

### Feature: DFT Interface (Mock)

```gherkin
Feature: Structure Labeling
  As a system
  I need to assign ground truth labels to selected structures
  So that I can train the final model

  Scenario: Labeling with Mock DFT
    Given a selected structure "B" with no energy/forces
    When I submit it to the DFT Manager (Mock)
    Then the structure "B" should be returned with a valid energy value
    And the structure "B" should have force vectors for all atoms
    And the status should be "LABELED"
```
