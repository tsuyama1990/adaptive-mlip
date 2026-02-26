# Cycle 03 UAT: MACE Surrogate Loop

## 1. Test Scenarios

### Scenario 3.1: MACE Fine-tuning
*   **ID**: UAT-03-01
*   **Priority**: High
*   **Description**: The system must successfully fine-tune a MACE model on the active set generated in Cycle 02.
*   **Pre-conditions**:
    *   `step2_active_set.xyz` exists and contains labeled structures.
    *   `config.yaml` specifies `step3_mace_finetune`.
*   **Post-conditions**:
    *   A new model file `work_dir/models/mace_finetuned.model` is created.
    *   The `workflow_state.json` is updated to point to this new model.
    *   Logs indicate training success (e.g., loss decreased or "Mock Training Complete").

### Scenario 3.2: Surrogate MD Generation
*   **ID**: UAT-03-02
*   **Priority**: High
*   **Description**: The system must run MD simulations using the fine-tuned MACE model to generate a large surrogate dataset.
*   **Pre-conditions**:
    *   `mace_finetuned.model` exists.
    *   `config.yaml` specifies `step4_surrogate_sampling`.
*   **Post-conditions**:
    *   A file `step4_surrogate_data.xyz` is created.
    *   It contains thousands of structures (as per config).
    *   The structures are physically valid (no overlaps, reasonable bond lengths).

## 2. Behavior Definitions (Gherkin)

### Feature: MACE Fine-tuning

```gherkin
Feature: Model Specialization
  As a researcher
  I want to fine-tune the MACE model on my specific system
  So that I can explore the PES accurately

  Scenario: Successful Fine-tuning
    Given a labeled dataset "step2_active_set.xyz"
    And a configuration with "epochs: 10"
    When I run the fine-tuning step
    Then the model "models/mace_finetuned.model" should be created
    And the training log should show "Epoch 10 completed"
```

### Feature: Molecular Dynamics Sampling

```gherkin
Feature: Phase Space Exploration
  As a researcher
  I want to run MD with the specialized model
  So that I can generate diverse configurations for the final potential

  Scenario: Generating Surrogate Data
    Given a fine-tuned MACE model
    And an MD configuration "temperature: 300K, steps: 1000"
    When I run the surrogate sampling step
    Then the trajectory file "step4_surrogate_data.xyz" should be created
    And the file should contain approximately 100 frames (assuming interval 10)
    And the average temperature of the trajectory should be near 300K
```
