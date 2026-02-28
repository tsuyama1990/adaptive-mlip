# CYCLE 01: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-01-01 (Priority: High)
**Title:** Zero-Shot Distillation Baseline Construction
**Description:** Verify that the system can automatically generate a physically sound baseline potential (`base.yace`) for a multi-component system entirely without invoking Quantum Espresso, relying strictly on the foundational MACE model's zero-shot capabilities and the new configuration schemas. This proves Phase 1 of the architecture is functional and respects the `DistillationConfig`.

### Scenario ID: UAT-01-02 (Priority: Medium)
**Title:** Intelligent Cutout and Auto-Passivation
**Description:** Ensure that the `extract_intelligent_cluster` utility correctly isolates a high-uncertainty region from a massive bulk structure, strictly applies `force_weight` arrays according to `CutoutConfig` radii, and crucially, auto-passivates any dangling bonds at the boundary to prevent electronic divergence in subsequent DFT calculations.

## 2. Behavior Definitions (Gherkin)

### UAT-01-01: Zero-Shot Distillation Baseline Construction

```gherkin
FEATURE: Zero-Shot Phase 1 Initialization
  As a materials computational scientist
  I want to generate a baseline ML potential without expensive DFT calls
  So that I can quickly start large-scale MD simulations with reasonable accuracy.

  SCENARIO: Generating a baseline potential for a 4-element alloy
    GIVEN a `config.yaml` with `DistillationConfig.enable = True`
    AND `elements = ["Fe", "Pt", "Mg", "O"]`
    AND a mocked `MACEManager` that returns deterministic uncertainties
    WHEN I initialize the Phase 1 workflow
    THEN the system automatically generates combinatorial sub-system structures
    AND DIRECT sampling reduces the pool size to `sampling_structures_per_system`
    AND the `MACEManager` filters out structures where uncertainty > `uncertainty_threshold`
    AND the system invokes the `PacemakerTrainer` to train a `base.yace`
    AND the `QEDriver` (DFT) is NEVER called during this entire process.
```

### UAT-01-02: Intelligent Cutout and Auto-Passivation

```gherkin
FEATURE: Intelligent Cluster Extraction
  As a system architect
  I want to extract a chemically stable cluster from a massive unstable MD frame
  So that DFT can converge quickly without encountering unphysical dangling bonds.

  SCENARIO: Extracting a defect core from bulk MgO
    GIVEN a massive (10,000 atom) bulk MgO structure with a single oxygen vacancy
    AND a `CutoutConfig` defining `core_radius = 4.0` and `buffer_radius = 3.0`
    AND `enable_passivation = True` with `passivation_element = "H"`
    WHEN I call `extract_intelligent_cluster` targeting the atoms adjacent to the vacancy
    THEN a new, smaller `Atoms` object is returned
    AND the `arrays['force_weight']` contains `1.0` ONLY for atoms within 4.0Å of the target
    AND the `arrays['force_weight']` contains `0.0` for atoms between 4.0Å and 7.0Å
    AND all other bulk atoms are deleted
    AND Hydrogen (H) atoms are automatically appended to under-coordinated Mg/O atoms at the boundary
    AND the newly added H atoms have `force_weight = 0.0`.
```