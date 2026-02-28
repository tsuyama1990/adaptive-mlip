# CYCLE 01: Core Extraction & Tiered Evaluation

## 1. Summary

This cycle lays the foundation for the "NextGen Hierarchical Distillation Architecture." The primary goal is to implement the "Intelligent Cutout & Passivation" logic (Phase 3 of the PRD) and the data structures necessary to configure it. We will introduce the new configuration schemas (`DistillationConfig`, `ActiveLearningThresholds`, `CutoutConfig`, `LoopStrategyConfig`) to the `domain_models` and build the `extraction.py` module. This module is responsible for identifying the "epicenter" of uncertainty, spherically extracting a localized cluster, applying force weights, performing boundary relaxation using a foundational model (MACE), and auto-passivating dangling bonds to ensure DFT convergence.

## 2. System Architecture

The scope of this cycle touches the core configuration models and introduces a new utility module for structural manipulation.

```text
src/pyacemaker/
├── core/
│   └── (No changes in this cycle)
├── domain_models/
│   ├── **config.py**           (Modify: Add new Pydantic models)
│   └── data.py
└── utils/
    ├── **extraction.py**       (Create: Intelligent Cutout logic)
    └── path.py
```

## 3. Design Architecture

This cycle relies heavily on robust Pydantic schemas to define the rules of engagement for the Active Learning loop and structural extraction.

### 3.1. Domain Models (`domain_models/config.py`)

We will extend `PyAceConfig` (or the relevant top-level config) to include the following nested models:

*   **`DistillationConfig`**: Configures Phase 1 (Zero-Shot Distillation).
    *   `enable` (bool, default `True`)
    *   `mace_model_path` (str, default `"mace-mp-0-medium"`)
    *   `uncertainty_threshold` (float, default `0.05`): The threshold below which MACE is considered "confident."
    *   `sampling_structures_per_system` (int, default `1000`)
*   **`ActiveLearningThresholds`**: Implements the FLARE-inspired two-tier threshold system.
    *   `threshold_call_dft` (float, default `0.05`): The system-wide max uncertainty required to halt MD.
    *   `threshold_add_train` (float, default `0.02`): The per-atom uncertainty required to include an atom in the local learning set (the epicenter).
    *   `smooth_steps` (int, default `3`): The number of consecutive steps the threshold must be exceeded to trigger a halt (thermal noise exclusion).
*   **`CutoutConfig`**: Configures Phase 3 (Intelligent Cutout).
    *   `core_radius` (float, default `4.0`): Radius in Angstroms around the epicenter where `force_weight=1.0`.
    *   `buffer_radius` (float, default `3.0`): Additional radius for the boundary layer (`force_weight=0.0`).
    *   `enable_pre_relaxation` (bool, default `True`): Whether to relax the buffer layer using MACE.
    *   `enable_passivation` (bool, default `True`): Whether to auto-passivate dangling bonds.
    *   `passivation_element` (str, default `"H"`): The element to use for passivation.
*   **`LoopStrategyConfig`**: Configures the overall active learning loop strategy.
    *   `use_tiered_oracle` (bool, default `True`)
    *   `incremental_update` (bool, default `True`)
    *   `replay_buffer_size` (int, default `500`)
    *   `baseline_potential_type` (str, default `"LJ"`)
    *   `thresholds` (ActiveLearningThresholds)

### 3.2. Extraction Module (`utils/extraction.py`)

This module exposes the core function: `extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: CutoutConfig) -> Atoms`.

**Key Invariants & Constraints:**
*   The input `structure` must be a valid ASE `Atoms` object with periodic boundary conditions.
*   The output cluster must retain the `force_weight` array in its `arrays` dictionary. Core atoms get `1.0`, buffer atoms get `0.0`, and passivating atoms get `0.0`.
*   During pre-relaxation (`_pre_relax_buffer`), core atoms MUST be strictly constrained using `ase.constraints.FixAtoms`.
*   Auto-passivation (`_passivate_surface`) must correctly identify under-coordinated atoms at the boundary based on covalent radii and add the specified `passivation_element` (e.g., Hydrogen) at a reasonable bond distance.

## 4. Implementation Approach

1.  **Define Pydantic Models:**
    *   Open `src/pyacemaker/domain_models/config.py`.
    *   Import necessary fields from `pydantic`.
    *   Define the four new configuration classes (`DistillationConfig`, `ActiveLearningThresholds`, `CutoutConfig`, `LoopStrategyConfig`).
    *   Ensure they are integrated into the main `PyAceConfig` schema (as optional fields to maintain backward compatibility if necessary, though this is a major version update).
2.  **Implement Extraction Logic:**
    *   Create `src/pyacemaker/utils/extraction.py`.
    *   Implement the main `extract_intelligent_cluster` function.
    *   **Step 1: Spherical Cutout:** Use ASE's `neighbor_list` or KDTree to find all atoms within `core_radius` of any atom in `target_atoms`. Assign them `force_weight=1.0`. Then, find all atoms within `core_radius + buffer_radius`. Assign the newly found atoms `force_weight=0.0`. Delete all other atoms.
    *   **Step 2: Pre-relaxation (Mocked MACE for now):** Create a helper function `_pre_relax_buffer(cluster, mace_calc_mock)`. Apply `FixAtoms` to the core atoms. Run an `LBFGS` optimization on the cluster using the provided calculator. (For this cycle, we assume the caller provides a valid ASE calculator, which will be the MACE wrapper in later cycles).
    *   **Step 3: Auto-Passivation:** Create `_passivate_surface(cluster, element)`. Identify boundary atoms (those with `force_weight=0.0`). Calculate their coordination number using covalent radii. If under-coordinated, calculate a normalized vector pointing outwards from the cluster center and add the `element` at a standard bond distance. Set `force_weight=0.0` for the new atoms.
3.  **Refine & Lint:** Run Ruff and MyPy to ensure strict typing and complexity limits are met.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)

Unit tests will focus on the isolation and correctness of the individual functions within `utils/extraction.py` and the validation logic of the new Pydantic models in `domain_models/config.py`.

1.  **Configuration Validation:**
    *   Test that default values are correctly assigned when instantiating `PyAceConfig` without the new sections.
    *   Test validation errors (e.g., providing negative radii in `CutoutConfig`, or invalid elements for passivation).
2.  **Spherical Cutout Logic (`extract_intelligent_cluster`):**
    *   Construct a simple cubic lattice (e.g., a 5x5x5 supercell of a single element).
    *   Select a central atom as the `target_atom`.
    *   Call the extraction function with a `core_radius` that should encompass the first neighbor shell and a `buffer_radius` for the second.
    *   **Assertions:** Verify the total number of atoms in the returned cluster matches the expected geometric count. Verify that the `arrays['force_weight']` exists and strictly contains `1.0` for core atoms and `0.0` for buffer atoms.
3.  **Pre-relaxation Constraints (`_pre_relax_buffer`):**
    *   Use a dummy ASE Calculator (like Lennard-Jones or EMT).
    *   Pass a cluster with predefined core and buffer atoms.
    *   Run the pre-relaxation step.
    *   **Assertions:** Check the positions of the core atoms before and after the LBFGS optimization. They must be bitwise identical. The buffer atoms' positions should have changed.
4.  **Auto-Passivation (`_passivate_surface`):**
    *   Create a small cluster with intentionally "broken" bonds at the surface (e.g., a silicon nanocluster).
    *   Run the passivation function.
    *   **Assertions:** Verify that new atoms (e.g., 'H') have been appended to the `Atoms` object. Verify their `force_weight` is `0.0`. Verify the distance between the passivating atom and its host is physically reasonable.

### Integration Testing Approach (Min 300 words)

Integration testing for this cycle will simulate the data flow from configuration parsing through to a complete extraction pipeline, ensuring the components interact correctly without external dependencies like real MACE or DFT.

1.  **End-to-End Extraction Pipeline:**
    *   Load a YAML configuration file containing the new `CutoutConfig` parameters.
    *   Instantiate the config model and parse the parameters.
    *   Generate a realistic, large-scale atomic structure (e.g., a 2000-atom MgO cell with a defect) using ASE.
    *   Simulate an "epicenter" by manually selecting a list of atom indices near the defect.
    *   Execute the full `extract_intelligent_cluster` function, passing a dummy fast calculator (e.g., `ase.calculators.lj.LennardJones`) to simulate the MACE pre-relaxation step.
    *   **Assertions:**
        *   The resulting cluster must be a valid, stand-alone ASE `Atoms` object.
        *   It must have a vacuum layer added (e.g., placed in a large unit cell to prevent periodic interactions, as it will be sent to DFT).
        *   The `force_weight` array must be contiguous and correctly assigned across core, buffer, and passivated atoms.
        *   The cluster must not contain overlapping atoms (a common artifact of automated passivation). Use `ase.neighborlist.neighbor_list` to assert no interatomic distances are below a physically absurd threshold (e.g., 0.5 Å).
2.  **Mocking the Calculator:** By using a fast, deterministic calculator like LJ or EMT during the integration test, we verify that the `extraction.py` module correctly interfaces with the standard ASE Calculator API, ensuring seamless integration with the actual `MACEManager` wrapper in Cycle 03.
