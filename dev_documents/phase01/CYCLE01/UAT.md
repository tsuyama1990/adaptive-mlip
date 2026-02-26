# Cycle 01 UAT: Foundation & Configuration

## 1. Test Scenarios

### Scenario 01-01: "Hello Config" (Priority: High)
**Objective**: Verify that the system can read a standard configuration file and start up without errors.
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 1)

1.  **Preparation**: Create a file `valid_config.yaml` with standard parameters (FePt alloy, 1000K, etc.).
2.  **Action**: Run `python -m pyacemaker.main --config valid_config.yaml --dry-run`.
3.  **Expectation**:
    *   Process exits with code 0.
    *   Console output shows: "Configuration loaded successfully."
    *   Console output shows: "Project: FePt_Optimization initialized."
    *   A log file `pyacemaker.log` is created.

### Scenario 01-02: "Guardrails Check" (Priority: Medium)
**Objective**: Verify that the system rejects physically impossible settings.

1.  **Preparation**: Create `bad_config.yaml` with `temperature: -100`.
2.  **Action**: Run `python -m pyacemaker.main --config bad_config.yaml`.
3.  **Expectation**:
    *   Process exits with non-zero code.
    *   Console output contains: "ValidationError: Temperature must be positive."

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Configuration Management

  Scenario: User provides a valid configuration
    GIVEN a YAML file "valid_config.yaml" located in current directory
    AND the file contains valid structure and DFT parameters
    WHEN I execute the command "pyacemaker run valid_config.yaml"
    THEN the system should validate the input schema
    AND the system should initialize the Orchestrator
    AND the logger should record "System initialized"

  Scenario: User provides an invalid configuration
    GIVEN a YAML file "bad_config.yaml"
    AND the file contains "cutoff_radius: -2.0"
    WHEN I execute the command "pyacemaker run bad_config.yaml"
    THEN the system should raise a Validation Error
    AND the error message should mention "cutoff_radius must be > 0"
    AND the system should not start the loop
```
