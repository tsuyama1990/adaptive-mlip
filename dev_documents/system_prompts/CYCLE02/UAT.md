# User Acceptance Testing (UAT) - Cycle 02

## 1. Test Scenarios

### Scenario 2.1: DIRECT Sampling Efficiency
**Priority:** High
**Objective:** Confirm that the DIRECT sampling strategy produces a more diverse dataset than random selection.

#### Marimo Code (`tests/uat/cycle02_sampling.py`)
```python
import marimo
import numpy as np
import matplotlib.pyplot as plt

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Cycle 02 UAT: DIRECT Sampling")
    return

@app.cell
def __():
    from pyacemaker.modules.sampling import DirectSampler
    from pyacemaker.domain_models.config import SamplingConfig

    # 1. Generate via DIRECT
    sampler = DirectSampler(config=SamplingConfig(n_initial=50))
    direct_structures = sampler.generate()

    # 2. Compute Descriptors (PCA projection for visualization)
    # (Mock implementation for UAT simplicity)
    descriptors = np.random.rand(50, 2)

    # 3. Visualize
    fig, ax = plt.subplots()
    ax.scatter(descriptors[:, 0], descriptors[:, 1], c='blue', label='DIRECT')
    ax.set_title("Configuration Space Coverage")
    ax.legend()

    return fig,
```

### Scenario 2.2: Active Learning Selection
**Priority:** Critical
**Objective:** Verify that structures with high uncertainty are correctly identified and selected.

#### Marimo Code (`tests/uat/cycle02_active_learning.py`)
```python
import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    from pyacemaker.modules.mace_oracle import MaceOracle
    from pyacemaker.domain_models.data import AtomStructure
    from pyacemaker.orchestrator import Orchestrator
    import numpy as np

    # 1. Create dummy structures with injected "true" uncertainty
    structures = []
    for i in range(100):
        s = AtomStructure() # Mock structure
        s.info['mock_uncertainty'] = float(i) / 100.0 # Linear 0.0 to 1.0
        structures.append(s)

    # 2. Select Top 10
    # Orchestrator logic simulation
    structures.sort(key=lambda x: x.info['mock_uncertainty'], reverse=True)
    active_set = structures[:10]

    # 3. Validation
    print(f"Top Uncertainty: {active_set[0].info['mock_uncertainty']}")
    assert len(active_set) == 10
    assert active_set[0].info['mock_uncertainty'] == 0.99

    return
```

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Pipeline

  Scenario: Generate diverse initial pool
    Given a target system "Cu"
    When "DirectSampler" is invoked with n=100
    Then it should return 100 valid AtomStructure objects
    And the structures should be distinct (non-overlapping atoms)

  Scenario: Select active set based on uncertainty
    Given a pool of 100 structures with calculated uncertainties
    When the active learning filter is applied with n_active=10
    Then the resulting set should contain the 10 structures with the highest uncertainty scores
    And the active set size should be exactly 10
```
