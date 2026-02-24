# Cycle 07 Specification: The Guardian (Validation & QA)

## 1. Summary
This cycle implements the **Guardian** module, responsible for the rigorous validation and quality assurance of the generated potentials. It integrates **Phonopy** to calculate phonon band structures (checking for dynamical stability via imaginary frequencies) and computes **Elastic Constants** (checking for mechanical stability via Born criteria). An **HTML Report Generator** is included to present these results in a user-friendly format, flagging unsafe potentials before they are deployed to production.

This cycle also addresses critical security, scalability, and maintainability issues identified in the previous audit.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/
│       │   ├── **validator.py**    # Validation Logic (Orchestrator)
│       │   └── **report.py**       # HTML Report Logic (Jinja2 + Matplotlib)
│       └── utils/
│           ├── **phonons.py**      # Phonopy Interface (Dynamical Stability)
│           └── **elastic.py**      # Elastic Constants Logic (Mechanical Stability)
└── tests/
    ├── **test_validator.py**
    ├── **test_phonons.py**
    ├── **test_elastic.py**
    └── **test_report.py**
```

## 3. Design Architecture

### 3.1 Domain Models (`domain_models/validation.py`, `config.py`)
*   **ValidationConfig**:
    *   `phonon_supercell`: list[int] (e.g., [2, 2, 2])
    *   `phonon_displacement`: float (e.g., 0.01)
    *   `elastic_strain`: float (e.g., 0.01)
    *   `imaginary_frequency_tolerance`: float (e.g., -0.05 THz)
    *   `symprec`: float (symmetry precision)
*   **ValidationResult**:
    *   `phonon_stable`: bool
    *   `elastic_stable`: bool
    *   `imaginary_frequencies`: list[float]
    *   `elastic_tensor`: list[list[float]]
    *   `bulk_modulus`: float
    *   `shear_modulus`: float
    *   `plots`: dict[str, str] (base64 encoded images)

### 3.2 Validator (`core/validator.py`)
Coordinates the validation suite.
*   **Input**: `PotentialPath`, `ValidationConfig`, `Structure`.
*   **Output**: `ValidationResult`.
*   **Logic**:
    1.  **Relaxation**: Minimize the structure using the potential before validation.
    2.  **Phonon Calculation**: Run `PhononCalculator`.
    3.  **Elastic Calculation**: Run `ElasticCalculator`.
    4.  **Stability Check**:
        *   Phonon: Check for imaginary modes < `imaginary_frequency_tolerance`.
        *   Elastic: Check Born stability criteria (generic or crystal-specific).
    5.  **Reporting**: Generate `ValidationResult` and call `ReportGenerator`.

### 3.3 Phonopy Interface (`utils/phonons.py`)
Wraps `phonopy`.
*   **Input**: `Atoms` (unit cell), `supercell_matrix`, `displacement_distance`.
*   **Output**: Band structure plot (base64), DOS plot (base64), max imaginary frequency.
*   **Logic**:
    *   Generate displacements.
    *   Calculate forces using `LammpsEngine` (or generic calculator).
    *   Compute force constants and band structure.

### 3.4 Elastic Constants (`utils/elastic.py`)
Calculates $C_{ij}$ matrix.
*   **Method**: Finite displacement (strain-stress).
*   **Logic**:
    *   Apply small strains ($\pm \epsilon$) to the unit cell.
    *   Relax atoms (optional) and compute stress tensor.
    *   Fit stress-strain curve to get $C_{ij}$.
    *   Compute Bulk (B) and Shear (G) moduli (Voigt-Reuss-Hill average).

### 3.5 Report Generator (`core/report.py`)
*   **Input**: `ValidationResult`.
*   **Output**: HTML string.
*   **Tech**: Jinja2 templating. Embeds plots as base64 strings.

## 4. Audit Fixes & Improvements

### 4.1 Security
*   **Path Whitelisting (`domain_models/dft.py`)**: `validate_pseudopotentials` must ensure paths are within allowed system directories or project scope.
*   **Command Injection (`utils/process.py`)**: `run_command` must use a strict allowlist for characters (alphanumeric, `_`, `-`, `.`, `/`, `=`, `,`, `:`, `+`, `@`). Error messages must sanitize the command to prevent leakage.

### 4.2 Scalability
*   **Active Set Selection (`core/active_set.py`)**: Implement chunked processing. Split large datasets into chunks (e.g., 10k frames), run selection on each, then merge and select again. This prevents `pace_activeset` (if it loads full files) from OOMing on huge datasets.
*   **Oracle OOM (`core/oracle.py`)**: `DFTManager.compute` must check the number of atoms in the embedded cluster. If `len(atoms) > 500` (configurable), raise `OracleError` or skip, to prevent DFT codes from crashing or hanging the node.

### 4.3 Maintainability
*   **LAMMPS Atom Style (`core/io_manager.py`)**: Remove hardcoded assumptions about atom style in fallback logic. Use `MDConfig.atom_style` consistently. Improve error logging to show the exact exception from the streaming writer.
*   **Magic Numbers**: Move hardcoded values (e.g., `base_energy = -100.0`) from `domain_models/md.py` to `defaults.py` or `constants.py`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_phonons.py`**: Mock `phonopy` and `ase.calculator`. Verify correct API calls and result parsing.
*   **`test_elastic.py`**: Mock stress-strain data. Verify $C_{ij}$ calculation and stability check for cubic/isotropic cases.
*   **`test_validator.py`**: Mock sub-components. Verify overall flow and result aggregation.
*   **`test_report.py`**: Verify HTML output contains expected sections and base64 strings.

### 5.2 UAT (`tests/uat/test_cycle07_uat.py`)
*   **Scenario 1**: Valid Potential. Run validation on a known stable potential (e.g., LJ argon). Expect PASS.
*   **Scenario 2**: Unstable Potential. Run validation on a bad potential (e.g., LJ with bad cutoff). Expect FAIL with "Imaginary Frequencies" or "Elastic Instability".
*   **Scenario 3**: Report Generation. Verify `validation_report.html` is created.

### 5.3 Coverage Goals
*   100% coverage on new modules.
*   Regression testing on modified core modules.
