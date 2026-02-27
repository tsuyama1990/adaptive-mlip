# User Acceptance Testing (UAT) - Cycle 01

## 1. Test Scenarios

### Scenario 1.1: System Initialization & Configuration
**Priority:** Critical
**Objective:** Verify that the system can read a configuration file and initialize the core data structures.

#### Marimo Code (`tests/uat/cycle01_init.py`)
```python
import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Cycle 01 UAT: Initialization")
    return

@app.cell
def __():
    from pyacemaker.domain_models.config import PyAceConfig, SystemConfig, DftConfig
    import yaml

    # 1. Create a dummy config
    config_dict = {
        "system": {"elements": ["Cu", "Zr"], "composition": {"Cu": 0.5, "Zr": 0.5}},
        "dft": {"code": "mock", "command": "python -m pyacemaker.mock"}
    }

    # 2. Validate via Pydantic
    config = PyAceConfig(**config_dict)
    print("Config Loaded Successfully:", config.system.elements)

    # 3. Assertions
    assert config.system.elements == ["Cu", "Zr"]
    assert config.dft.code == "mock"
    return config,
```

### Scenario 1.2: Mock Oracle Execution
**Priority:** High
**Objective:** Verify that the Mock Oracle can compute energy and forces for a simple structure.

#### Marimo Code (`tests/uat/cycle01_oracle.py`)
```python
import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    from pyacemaker.modules.mock_oracle import MockOracle
    from pyacemaker.domain_models.data import AtomStructure
    from ase.build import bulk

    # 1. Create a bulk structure
    atoms = bulk("Cu", "fcc", a=3.6)
    structure = AtomStructure.from_ase(atoms)

    # 2. Compute using MockOracle
    oracle = MockOracle()
    result = oracle.compute(structure)

    # 3. Verify Results
    print(f"Energy: {result.energy}")
    print(f"Forces:\n{result.forces}")

    assert result.energy is not None
    assert result.forces is not None
    return
```

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: System Initialization

  Scenario: Load valid configuration
    Given a configuration file "config.yaml" with elements ["Cu", "Zr"]
    When the "PyAceConfig" model loads the file
    Then the "system.elements" field should be ["Cu", "Zr"]
    And no validation errors should be raised

  Scenario: Compute Mock Energy
    Given a "MockOracle" instance
    And an "AtomStructure" representing bulk Copper
    When "compute" is called
    Then the returned structure should have a non-null "energy" field
    And the returned structure should have a "forces" array of shape (N, 3)
```
