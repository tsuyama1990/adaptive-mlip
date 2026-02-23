# Cycle 07 Specification: The Guardian (Validation & QA)

## 1. Summary
This cycle implements the **Guardian** module, responsible for the rigorous validation and quality assurance of the generated potentials. It integrates **Phonopy** to calculate phonon band structures (checking for dynamical stability via imaginary frequencies) and computes **Elastic Constants** (checking for mechanical stability via Born criteria). An **HTML Report Generator** is included to present these results in a user-friendly format, flagging unsafe potentials before they are deployed to production.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/
│       │   ├── **validator.py**    # Validation Logic
│       │   └── **report.py**       # HTML Report Logic
│       └── utils/
│           ├── **phonons.py**      # Phonopy Interface
│           └── **elastic.py**      # Elastic Constants Logic
└── tests/
    ├── **test_validator.py**
    ├── **test_phonons.py**
    └── **test_elastic.py**
```

## 3. Design Architecture

### 3.1 Validator (`core/validator.py`)
Coordinates the validation suite.
*   **Input**: `PotentialPath`, `ValidationConfig`.
*   **Output**: `ValidationReport` (HTML/JSON).
*   **Logic**:
    1.  Run Phonon Calculation.
    2.  Run Elastic Constant Calculation.
    3.  Check Criteria (Imaginary Freqs < Tolerance, Born Stability).
    4.  Return `PASS` / `FAIL`.

### 3.2 Phonopy Interface (`utils/phonons.py`)
Wraps `phonopy`.
*   **Input**: `Atoms` (unit cell), Supercell Matrix (e.g., [2,2,2]).
*   **Output**: Band Structure Plot, DOS Plot.

### 3.3 Elastic Constants (`utils/elastic.py`)
Calculates $C_{ij}$ matrix.
*   **Method**: Finite displacement (strain-stress).
*   **Logic**: Apply small strains ($\pm 1\%$), fit stress-strain curve.

## 4. Implementation Approach

### Step 1: Phonopy Integration
*   Implement `src/pyacemaker/utils/phonons.py`.
*   Ensure it can generate band structure data from ASE Calculator.

### Step 2: Elastic Constants
*   Implement `src/pyacemaker/utils/elastic.py`.
*   Implement Born Stability Criteria checks (e.g., $C_{11} - C_{12} > 0$).

### Step 3: Reporting
*   Implement `src/pyacemaker/core/report.py`.
*   Generate a static HTML file with plots (Matplotlib/Plotly) and tables.

## 5. Test Strategy

### 5.1 Unit Testing (`test_phonons.py`)
*   **Stable Crystal**: Calculate phonons for a perfect LJ crystal (fcc). Assert no imaginary modes.
*   **Unstable Crystal**: Create an artificially unstable structure (e.g., compressed beyond limit). Assert imaginary modes exist.

### 5.2 Unit Testing (`test_elastic.py`)
*   **Isotropic Material**: Verify $C_{11} \approx C_{12} + 2C_{44}$ for an isotropic solid (if applicable).
*   **Consistency**: Ensure calculated bulk modulus matches EOS fit.

### 5.3 Coverage Goals
*   100% coverage on stability criteria logic.
*   Verify HTML generation (file exists, contains specific strings).
