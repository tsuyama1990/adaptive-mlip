# Cycle 08 Specification: The Expander (kMC & Production)

## 1. Summary
This final cycle extends the system capabilities to **Adaptive Kinetic Monte Carlo (aKMC)** using the **EON** software, allowing the simulation of long-timescale phenomena like diffusion and phase ordering. It also focuses on the **Production Readiness** of the system, including the implementation of the specific **Fe/Pt on MgO** deposition scenario ("Grand Challenge") and comprehensive documentation generation.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── interfaces/
│       │   └── **eon_driver.py**   # EON Interface
│       └── scenarios/
│           ├── **fept_mgo.py**     # Grand Challenge Logic
│           └── **base_scenario.py** # (New)
└── tests/
    ├── **test_eon.py**
    └── **test_scenarios.py**
```

## 3. Design Architecture

### 3.1 EON Wrapper (`interfaces/eon_driver.py`)
Manages the EON execution.
*   **Input**: `ReactionConfig`, `PotentialPath`.
*   **Output**: `ReactionEvent` (barrier, product state).
*   **Logic**:
    *   Generates `config.ini` for EON.
    *   Launches `eonclient`.
    *   Parses `dynamics.txt` / `processtable.dat`.

### 3.2 Scenario Logic (`scenarios/fept_mgo.py`)
Encapsulates the complexity of the specific user story.
*   **Steps**:
    1.  Generate MgO (001) Surface.
    2.  Deposit Fe/Pt atoms randomly (MD).
    3.  Run aKMC to observe L10 ordering.
    4.  Visualize.

## 4. Implementation Approach

### Step 1: EON Interface
*   Implement `src/pyacemaker/interfaces/eon_driver.py`.
*   Create a "PACE Driver" script that EON can call to evaluate energies (via stdin/stdout).

### Step 2: Scenario Scripts
*   Implement `src/pyacemaker/scenarios/fept_mgo.py`.
*   Use `ase` to build the initial slab.
*   Use `MDEngine` for deposition.
*   Use `EONWrapper` for ordering.

### Step 3: Final Polish
*   Update `README.md` with final instructions.
*   Ensure all docstrings are complete.

## 5. Test Strategy

### 5.1 Unit Testing (`test_eon.py`)
*   **Config Gen**: Verify `config.ini` is correctly formatted.
*   **Driver Comm**: Test the stdin/stdout communication protocol for the PACE driver script.

### 5.2 Integration Testing (`test_scenarios.py`)
*   **End-to-End**: Run the `FePt_MgO` scenario in "Mock Mode".
    *   Verify all steps execute.
    *   Verify outputs (trajectory, process table) are created.

### 5.3 Coverage Goals
*   100% coverage on EON config generation.
*   Scenario script execution (mocked).
