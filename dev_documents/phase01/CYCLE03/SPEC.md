# Cycle 03 Specification: The Explorer (Structure Generation)

## 1. Summary
This cycle implements the **Structure Generator**, the engine responsible for exploring the vast chemical and structural space to propose new atomic configurations. It integrates with **M3GNet** (or a mock) to provide "Cold Start" structures and employs sophisticated perturbation policies (Random Displacement, Strain, Defects) to generate diverse candidates for Active Learning.

## 2. System Architecture
```ascii
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/
│       │   ├── **generator.py**    # Structure Generation Logic
│       │   ├── **policy.py**       # Exploration Strategies
│       │   └── **m3gnet_wrapper.py** # Initial Guess
│       └── utils/
│           └── **perturbations.py** # Rattling/Straining/Defects
└── tests/
    ├── **test_generator.py**
    └── **test_policy.py**
```

## 3. Design Architecture

### 3.1 Structure Generator (`core/generator.py`)
This class manages the generation pipeline. It uses the Strategy Pattern to select the appropriate `ExplorationPolicy`.
*   **Input**: `Composition` (e.g., "FePt"), `ExplorationConfig`.
*   **Output**: `List[Atoms]`.

### 3.2 Exploration Policies (`core/policy.py`)
We implement several policies as subclasses of `BasePolicy`:
*   `ColdStartPolicy`: Uses M3GNet to find stable bulk structures.
*   `RattlePolicy`: Applies random Gaussian noise to atomic positions.
*   `StrainPolicy`: Applies random strain tensors (volume change, shear).
*   `DefectPolicy`: Creates vacancies or interstitials.

### 3.3 Perturbation Utils (`utils/perturbations.py`)
Helper functions to manipulate `ase.Atoms` objects safely.
*   `rattle(atoms, stdev)`
*   `apply_strain(atoms, strain_tensor)`
*   `create_vacancy(atoms, num_vacancies)`

## 4. Implementation Approach

### Step 1: M3GNet Integration
*   Implement `src/pyacemaker/core/m3gnet_wrapper.py`.
*   Allow it to predict relaxed structures given a composition string.

### Step 2: Perturbation Logic
*   Implement `src/pyacemaker/utils/perturbations.py`.
*   Verify that stoichiometry is preserved (except for defects).

### Step 3: Generator Orchestration
*   Implement `src/pyacemaker/core/generator.py`.
*   Connect the policy selection logic based on `config.yaml`.

## 5. Test Strategy

### 5.1 Unit Testing (`test_generator.py`)
*   **Stoichiometry Check**: Verify that `RattlePolicy` does not change the number of atoms.
*   **Lattice Check**: Verify that `StrainPolicy` changes the cell volume correctly.
*   **Defect Check**: Verify that `DefectPolicy` reduces the number of atoms (for vacancies).

### 5.2 Integration Testing (`test_policy.py`)
*   **Policy Selection**: Ensure the correct policy is instantiated based on the config string.
*   **Pipeline**: Ensure generated structures are valid ASE objects and can be written to file (e.g., XYZ).

### 5.3 Coverage Goals
*   100% coverage on perturbation functions.
*   Mock M3GNet calls to test the wrapper without heavy dependencies.
