# Cycle 01 UAT: Core Framework & DIRECT Sampling

## 1. Test Scenarios

### Scenario 1.1: Project Initialization
*   **ID**: UAT-01-01
*   **Priority**: Critical
*   **Description**: The system must successfully initialize the workspace from a configuration file.
*   **Pre-conditions**: A valid `config.yaml` file exists.
*   **Post-conditions**:
    *   A working directory (e.g., `./work`) is created.
    *   A `workflow_state.json` file is created, indicating the initial state (PENDING).
    *   Logs are written to `pyacemaker.log`.

### Scenario 1.2: DIRECT Sampling Execution
*   **ID**: UAT-01-02
*   **Priority**: High
*   **Description**: The system generates a diverse set of initial structures using the DIRECT sampling method.
*   **Pre-conditions**: `config.yaml` specifies `target_points: 10` and `elements: ["Si"]`.
*   **Post-conditions**:
    *   A file named `step1_initial.xyz` is created in the working directory.
    *   The file contains exactly 10 valid atomic structures.
    *   No two atoms in any structure are closer than 1.5 Angstroms (hard-sphere check).
    *   The workflow state is updated to reflect Step 1 completion.

### Scenario 1.3: Invalid Configuration Handling
*   **ID**: UAT-01-03
*   **Priority**: Medium
*   **Description**: The system must gracefully reject invalid configurations.
*   **Pre-conditions**: `config.yaml` contains `target_points: -5`.
*   **Post-conditions**:
    *   The program exits with a non-zero status code.
    *   A clear error message is displayed (e.g., "Target points must be positive").
    *   No working directory is created or modified.

## 2. Behavior Definitions (Gherkin)

### Feature: Workspace Initialization

```gherkin
Feature: Workspace Setup
  As a user
  I want to initialize a new project
  So that I can start the potential generation workflow

  Scenario: Successful Initialization
    Given a configuration file "config.yaml" with valid parameters
    When I run the command "pyacemaker init --config config.yaml"
    Then a directory "./work" should be created
    And a file "./work/workflow_state.json" should exist
    And the log file should contain "Workspace initialized"
```

### Feature: DIRECT Sampling

```gherkin
Feature: Structure Generation
  As a user
  I want to generate initial structures
  So that I have a diverse starting point for active learning

  Scenario: Generating 10 Silicon Structures
    Given a configuration file "config.yaml" specifying:
      | parameter      | value |
      | target_points  | 10    |
      | elements       | ["Si"]|
    When I run the command "pyacemaker run --step 1"
    Then the file "./work/step1_initial.xyz" should be created
    And the file should contain 10 frames
    And all frames should contain only "Si" atoms
    And no atom pair distance should be less than 1.5 Angstroms
```
