# Cycle 08 Specification: The Expander (kMC & Production)

## 1. Summary
This final cycle extends the system capabilities to **Adaptive Kinetic Monte Carlo (aKMC)** using the **EON** software, allowing the simulation of long-timescale phenomena like diffusion and phase ordering. It also focuses on the **Production Readiness** of the system, including the implementation of the specific **Fe/Pt on MgO** deposition scenario ("Grand Challenge") and comprehensive documentation generation.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── domain_models/
│       │   ├── **eon.py**          # EON Config Model
│       │   └── **scenario.py**     # Scenario Config Model
│       ├── interfaces/
│       │   └── **eon_driver.py**   # EON Interface
│       └── scenarios/
│           ├── **fept_mgo.py**     # Grand Challenge Logic
│           └── **base_scenario.py** # (New)
└── tests/
    ├── **test_domain_models_eon.py**
    ├── **test_domain_models_scenario.py**
    ├── **test_eon_driver.py**
    └── **test_scenarios.py**
```

## 3. Design Architecture

### 3.1 Domain Models
*   **EONConfig (`domain_models/eon.py`)**: Defines configuration for EON simulations (e.g., temperature, process search parameters).
*   **ScenarioConfig (`domain_models/scenario.py`)**: Defines configuration for specific scenarios (e.g., selection of `fept_mgo`).

### 3.2 EON Wrapper (`interfaces/eon_driver.py`)
Manages the EON execution.
*   **Input**: `EONConfig`, `PotentialPath`.
*   **Output**: `ReactionEvent` (barrier, product state) or Process Table.
*   **Logic**:
    *   Generates `config.ini` for EON.
    *   Generates `pace_driver.py` script for EON to call.
    *   Launches `eonclient`.
    *   Parses `dynamics.txt` / `processtable.dat`.

### 3.3 Scenario Logic (`scenarios/fept_mgo.py`)
Encapsulates the complexity of the specific user story.
*   **Input**: `PyAceConfig` (containing EON and Scenario settings).
*   **Steps**:
    1.  Generate MgO (001) Surface.
    2.  Deposit Fe/Pt atoms randomly (MD using `LammpsEngine`).
    3.  Run aKMC to observe L10 ordering (using `EONWrapper`).
    4.  Visualize/Report.

## 4. Implementation Approach

### Step 1: Domain Models
*   Implement `src/pyacemaker/domain_models/eon.py`.
*   Implement `src/pyacemaker/domain_models/scenario.py`.
*   Update `src/pyacemaker/domain_models/config.py`.

### Step 2: EON Interface
*   Implement `src/pyacemaker/interfaces/eon_driver.py`.
*   Create a "PACE Driver" script generator that EON can call to evaluate energies (via stdin/stdout).

### Step 3: Scenario Scripts
*   Implement `src/pyacemaker/scenarios/base_scenario.py`.
*   Implement `src/pyacemaker/scenarios/fept_mgo.py`.
*   Use `ase` to build the initial slab.
*   Use `LammpsEngine` for deposition.
*   Use `EONWrapper` for ordering.

### Step 4: Main Integration
*   Update `src/pyacemaker/main.py` to handle `--scenario` flag.

### Step 5: Final Polish
*   Update `README.md` with final instructions.
*   Ensure all docstrings are complete.

## 5. Test Strategy

### 5.1 Unit Testing (`test_eon_driver.py`)
*   **Config Gen**: Verify `config.ini` is correctly formatted.
*   **Driver Comm**: Test the stdin/stdout communication protocol for the PACE driver script.

### 5.2 Integration Testing (`test_scenarios.py`)
*   **End-to-End**: Run the `FePt_MgO` scenario in "Mock Mode".
    *   Verify all steps execute.
    *   Verify outputs (trajectory, process table) are created.

### 5.3 Coverage Goals
*   100% coverage on EON config generation.
*   Scenario script execution (mocked).
