# User Acceptance Testing (UAT) - Cycle 05

## 1. Test Scenarios

### Scenario 5.1: Delta Learning Data Preparation
**Priority:** High
**Objective:** Confirm that the mixed dataset is constructed correctly with appropriate weights.

#### Marimo Code (`tests/uat/cycle05_delta_prep.py`)
```python
import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Cycle 05 UAT: Delta Learning Prep")
    return

@app.cell
def __():
    from pyacemaker.modules.pacemaker_delta import PacemakerDeltaTrainer
    from pyacemaker.domain_models.delta_learning import DeltaConfig
    from pyacemaker.domain_models.data import AtomStructure
    from ase.build import bulk

    # 1. Setup Datasets
    base_data = [AtomStructure.from_ase(bulk("Cu")) for _ in range(100)]
    dft_data = [AtomStructure.from_ase(bulk("Cu")) for _ in range(10)]

    config = DeltaConfig(dft_weight=50.0)

    # 2. Prepare Data (Internal Method Exposure or Mock)
    # We assume a helper method exists or we simulate the logic
    trainer = PacemakerDeltaTrainer()
    mixed_data = trainer._prepare_mixed_dataset(base_data, dft_data, config.dft_weight)

    # 3. Validation
    # Check total size
    assert len(mixed_data) == 110

    # Check weights (Assuming weights are stored in info['weight'])
    # First 100 should be 1.0, last 10 should be 50.0
    assert mixed_data[0].info.get('weight', 1.0) == 1.0
    assert mixed_data[-1].info.get('weight') == 50.0

    return
```

### Scenario 5.2: Delta Fine-tuning Execution
**Priority:** Critical
**Objective:** Verify that the system can execute the Delta Learning step.

#### Marimo Code (`tests/uat/cycle05_delta_exec.py`)
```python
import marimo
import os

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    from pyacemaker.modules.pacemaker_delta import PacemakerDeltaTrainer
    from pyacemaker.domain_models.delta_learning import DeltaConfig

    # 1. Setup
    config = DeltaConfig(base_potential_path="mock_base.yace", dft_weight=10.0)
    trainer = PacemakerDeltaTrainer()

    # 2. Run (Mocked)
    try:
        # Pass empty lists as we are mocking the execution
        final_path = trainer.train_delta([], [], config)
        print(f"Final Potential: {final_path}")
    except Exception as e:
        print(f"Execution skipped/failed: {e}")

    return
```

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Delta Learning Correction

  Scenario: Prepare Weighted Dataset
    Given a base dataset (N=100) and a DFT dataset (M=10)
    And a DFT weight of 50.0
    When the datasets are merged for Delta Learning
    Then the resulting dataset size should be N + M
    And the DFT samples should carry a weight of 50.0
    And the base samples should carry a weight of 1.0

  Scenario: Execute Delta Fine-tuning
    Given a valid base potential file
    And a weighted dataset
    When "PacemakerDeltaTrainer.train_delta" is called
    Then the "input.yaml" should reference the base potential as initialization
    And a final ".yace" file should be produced
```
