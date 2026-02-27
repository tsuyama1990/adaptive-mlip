# User Acceptance Testing (UAT) - Cycle 06

## 1. Test Scenarios

### Scenario 6.1: Full Pipeline Execution (Mock Mode)
**Priority:** Critical
**Objective:** Verify that the entire 7-step pipeline runs from start to finish without error in Mock Mode.

#### Marimo Code (`tutorials/UAT_AND_TUTORIAL.py` - Excerpt)
```python
import marimo
import os

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# PyAceMaker: End-to-End Tutorial")
    return

@app.cell
def __():
    import os
    # Force Mock Mode for CI/Tutorial
    os.environ["PYACEMAKER_MODE"] = "MOCK"

    from pyacemaker.main import main
    # Simulate CLI arguments
    import sys
    sys.argv = ["pyacemaker", "run", "--config", "examples/tutorial_config.yaml"]

    # 1. Run the pipeline
    try:
        main()
        success = True
    except Exception as e:
        success = False
        print(f"Pipeline Failed: {e}")

    # 2. Verify Artifacts
    assert success
    assert os.path.exists("models/final.yace")
    return
```

### Scenario 6.2: Resume Capability
**Priority:** Medium
**Objective:** Verify that the pipeline can resume from a saved state.

#### Marimo Code (`tests/uat/cycle06_resume.py`)
```python
import marimo
import json
import os

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    from pyacemaker.orchestrator import Orchestrator

    # 1. Create a dummy state file indicating Step 3 is done
    state = {"step": 3, "artifacts": {"mace_model": "mock.model"}}
    with open("state.json", "w") as f:
        json.dump(state, f)

    # 2. Initialize Orchestrator with resume=True
    orch = Orchestrator(resume=True)

    # 3. Verify it attempts Step 4
    assert orch.current_step == 4
    return
```

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: End-to-End Workflow

  Scenario: Run Full Pipeline
    Given a valid configuration file
    And "PYACEMAKER_MODE" set to "MOCK"
    When "pyacemaker run" is executed
    Then the process should exit with code 0
    And the file "models/final.yace" should exist

  Scenario: Resume from Checkpoint
    Given a "state.json" file indicating completion of Step 3
    When "pyacemaker run --resume" is executed
    Then the pipeline should skip Steps 1-3
    And execution should begin at Step 4
```
