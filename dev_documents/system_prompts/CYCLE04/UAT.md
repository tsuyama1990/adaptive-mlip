# Cycle 04 UAT: Surrogate Labeling & Base ACE Training

## 1. Test Scenarios

### Scenario 4.1: Batch Surrogate Labeling
*   **ID**: UAT-04-01
*   **Priority**: High
*   **Description**: The system must label thousands of structures efficiently using the fine-tuned MACE model.
*   **Pre-conditions**:
    *   `step4_surrogate_data.xyz` (Unlabeled) exists.
    *   `mace_finetuned.model` exists.
*   **Post-conditions**:
    *   `step5_surrogate_labeled.xyz` is created.
    *   All frames contain `energy` (scalar) and `forces` (N x 3 vectors).
    *   The `provenance` metadata is "MACE_SURROGATE".

### Scenario 4.2: Base ACE Training (Pacemaker)
*   **ID**: UAT-04-02
*   **Priority**: Critical
*   **Description**: The system must successfully train a base ACE potential using the labeled surrogate dataset.
*   **Pre-conditions**:
    *   `step5_surrogate_labeled.xyz` exists.
    *   `config.yaml` specifies `step6_pacemaker_base`.
*   **Post-conditions**:
    *   A valid ACE potential file `work_dir/models/ace_base.yace` is created.
    *   The `workflow_state.json` is updated.
    *   Logs indicate training convergence (or successful execution in Mock mode).

## 2. Behavior Definitions (Gherkin)

### Feature: MACE Batch Labeling

```gherkin
Feature: Efficient Dataset Generation
  As a researcher
  I want to label my large MD trajectory quickly
  So that I can train the ACE potential without waiting weeks for DFT

  Scenario: Labeling 1000 Structures
    Given a trajectory file "step4_surrogate_data.xyz" with 1000 frames
    And a fine-tuned MACE model
    When I run the surrogate labeling step
    Then the output file "step5_surrogate_labeled.xyz" should contain 1000 frames
    And every frame should have valid Energy and Force properties
    And the processing time should be less than 5 minutes (assuming GPU)
```

### Feature: ACE Base Training

```gherkin
Feature: Initial Potential Construction
  As a researcher
  I want to create a fast ACE potential that mimics the MACE model
  So that I have a solid starting point for delta learning

  Scenario: Training the Base Model
    Given a labeled dataset "step5_surrogate_labeled.xyz"
    And an ACE configuration with "cutoff: 5.0"
    When I run the base training step
    Then the potential file "models/ace_base.yace" should be generated
    And the Pacemaker input file "input.yaml" should reflect the cutoff of 5.0
```
