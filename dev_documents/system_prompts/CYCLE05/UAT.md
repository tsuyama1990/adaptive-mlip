# Cycle 05 UAT: Delta Learning

## 1. Test Scenarios

### Scenario 5.1: Delta Training with Prior Potential
*   **ID**: UAT-05-01
*   **Priority**: Critical
*   **Description**: The system must load the base ACE potential and fine-tune it using DFT data.
*   **Pre-conditions**:
    *   `models/ace_base.yace` exists.
    *   `step2_active_set.xyz` (DFT) exists.
    *   `config.yaml` specifies `step7_delta_learning`.
*   **Post-conditions**:
    *   `final_potential.yace` is created.
    *   Pacemaker logs indicate "Loaded potential from: ...".
    *   The basis set size/shape is identical to the base potential.

### Scenario 5.2: DFT Accuracy Improvement
*   **ID**: UAT-05-02
*   **Priority**: High
*   **Description**: The final potential should show lower error on the DFT dataset compared to the base potential.
*   **Pre-conditions**:
    *   A test set of DFT data exists.
    *   Both Base and Final potentials are available.
*   **Post-conditions**:
    *   RMSE(Final, DFT) < RMSE(Base, DFT).
    *   (In Mock mode, simply check that the training process completed without error).

## 2. Behavior Definitions (Gherkin)

### Feature: Delta Fine-tuning

```gherkin
Feature: Accuracy Refinement
  As a researcher
  I want to incorporate high-accuracy DFT data into my potential
  So that I correct the systematic errors of the surrogate model

  Scenario: Running Delta Learning
    Given a base ACE potential "models/ace_base.yace"
    And a DFT dataset "step2_active_set.xyz"
    And a delta configuration with "dft_weight: 100.0"
    When I run the delta learning step
    Then the final potential "final_potential.yace" should be saved
    And the training should start from the base potential parameters
```

### Feature: Weighting Strategy

```gherkin
Feature: Data Imbalance Handling
  As a system
  I need to balance the large surrogate dataset and small DFT dataset
  So that the model learns the correct physics without overfitting

  Scenario: Dataset Preparation
    Given 1000 surrogate structures and 10 DFT structures
    When I prepare the mixed dataset for delta learning
    Then the DFT structures should have a weight of 100.0
    And the surrogate structures should have a weight of 0.1 (or as configured)
    And the resulting "step7_mixed_data.pdata" should contain 1010 frames
```
