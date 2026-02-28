# CYCLE 03: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-03-01 (Priority: Critical)
**Title:** Hierarchical Finetuning and Delta Update
**Description:** Verify that the system can successfully take a single, highly accurate DFT calculation of a defective cluster, use MACE to instantly generate hundreds of surrogate structures around it, and then perform a fast "Delta Learning" update (O(1) cost) on the existing ACE potential using a Replay Buffer to prevent catastrophic forgetting.

### Scenario ID: UAT-03-02 (Priority: High)
**Title:** Tiered Oracle Evaluation
**Description:** Ensure the `TieredOracle` correctly routes structure evaluations. Structures with low uncertainty should be handled instantly by the `MACEManager`, while structures exceeding the threshold should trigger the expensive `QEDriver` (DFT).

## 2. Behavior Definitions (Gherkin)

### UAT-03-01: Hierarchical Finetuning and Delta Update

```gherkin
FEATURE: Incremental Update and Replay Buffer
  As a system architect
  I want the system to learn incrementally without forgetting past data
  So that I can scale to millions of MD steps without O(N^2) training time explosions.

  SCENARIO: Fast potential update via Delta Learning
    GIVEN a `base.yace` potential and a `training_history.extxyz` file containing 10,000 structures
    AND a `LoopStrategyConfig` with `incremental_update = True` and `replay_buffer_size = 500`
    AND a single newly evaluated DFT cluster structure (the "epicenter")
    WHEN the `MACEManager` generates 50 randomized "surrogate" perturbations of the cluster
    AND the `IncrementalTrainer` is invoked with the 51 new structures
    THEN the `IncrementalTrainer` automatically selects the previous `base.yace` as the `initial_potential`
    AND the trainer samples exactly 449 structures from the `training_history` (totalling 500)
    AND the generated `input.yaml` configures Pacemaker for Delta Learning (optimizing only the difference)
    AND the resulting `current.yace` successfully completes a brief 1-epoch test compilation.
```

### UAT-03-02: Tiered Oracle Evaluation

```gherkin
FEATURE: Smart Evaluator Routing
  As a materials computational scientist
  I want to only run DFT when absolutely necessary
  So that I save thousands of CPU hours by using the foundational model wherever possible.

  SCENARIO: Routing an uncertain structure to DFT
    GIVEN a `TieredOracle` configured with a `MACEManager` and a `QEDriver`
    AND a `threshold_call_dft = 0.05`
    AND a generated structure (e.g., heavily strained lattice)
    WHEN the `TieredOracle` evaluates the structure
    AND the `MACEManager`'s initial pass returns an uncertainty of 0.08
    THEN the `TieredOracle` MUST automatically invoke the `QEDriver`
    AND return the exact `energy` and `forces` calculated by Quantum Espresso
    AND the `MACEManager`'s forces are discarded in favor of the DFT ground truth.
```