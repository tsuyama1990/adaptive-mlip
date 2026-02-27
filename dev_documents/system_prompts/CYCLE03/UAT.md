# User Acceptance Testing (UAT) - Cycle 03

## 1. Test Scenarios

### Scenario 3.1: MACE Fine-tuning
**Priority:** Critical
**Objective:** Confirm that the system can initiate and complete a fine-tuning run on the MACE model.

#### Marimo Code (`tests/uat/cycle03_finetune.py`)
```python
import marimo
import os

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Cycle 03 UAT: MACE Fine-tuning")
    return

@app.cell
def __():
    from pyacemaker.modules.mace_trainer import MaceTrainer
    from pyacemaker.domain_models.training import MaceTrainingConfig
    from pyacemaker.domain_models.data import AtomStructure

    # 1. Setup Mock Data
    dataset = [AtomStructure() for _ in range(10)]
    config = MaceTrainingConfig(epochs=1, base_model="mock") # Use mock mode

    # 2. Run Trainer
    trainer = MaceTrainer()
    model_path = trainer.train(dataset, config)

    # 3. Validation
    print(f"Model saved to: {model_path}")
    assert os.path.exists(model_path)
    return
```

### Scenario 3.2: Surrogate MD Generation
**Priority:** High
**Objective:** Verify that the fine-tuned model can drive stable MD simulations to generate new structures.

#### Marimo Code (`tests/uat/cycle03_md.py`)
```python
import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    from pyacemaker.modules.md_generator import MdGenerator
    from pyacemaker.domain_models.training import MdConfig
    from pyacemaker.modules.mock_oracle import MockOracle
    from ase.build import bulk
    from pyacemaker.domain_models.data import AtomStructure

    # 1. Setup
    initial_atoms = [AtomStructure.from_ase(bulk("Cu"))]
    config = MdConfig(temperature=300, steps=50)

    # 2. Run MD (using MockOracle as a stand-in for the fine-tuned model)
    # In real run, we'd pass the model path. Here we pass a mock calculator.
    generator = MdGenerator(calculator=MockOracle().get_ase_calculator())
    surrogate_pool = generator.generate(n_structures=20, initial=initial_atoms, config=config)

    # 3. Validation
    print(f"Generated {len(surrogate_pool)} structures.")
    assert len(surrogate_pool) == 20
    # Check for non-zero displacements
    assert surrogate_pool[-1].atoms.positions[0, 0] != initial_atoms[0].atoms.positions[0, 0]
    return
```

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Surrogate Model Refinement

  Scenario: Fine-tune MACE model
    Given a labelled dataset of 10 structures
    And a valid MACE configuration
    When "MaceTrainer.train" is executed
    Then a ".model" file should be created
    And the training log should indicate completion

  Scenario: Generate surrogate data via MD
    Given a valid interatomic potential model
    And an initial structure
    When "MdGenerator.generate" runs for 50 steps
    Then it should return a list of structures sampled from the trajectory
    And the structures should have valid energies and forces
```
