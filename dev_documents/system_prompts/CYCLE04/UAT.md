# Cycle 04 UAT: The Trainer (Pacemaker Integration)

## 1. Test Scenarios

### Scenario 04-01: "Fit Potential" (Priority: High)
**Objective**: Verify that the system can take a dataset and produce a potential file.
**Marimo File**: `tutorials/UAT_AND_TUTORIAL.py` (Section 3 - Trainer)

1.  **Preparation**:
    *   Create a dummy dataset `train.pckl.gzip` (using Pacemaker format or ASE -> Pacemaker converter).
    *   Set `config.yaml` with `potential_type: ace` and `baseline: lj`.
2.  **Action**: Run `Trainer.fit(dataset="train.pckl.gzip")`.
3.  **Expectation**:
    *   Process completes without error.
    *   A file `output_potential.yace` is created.
    *   Log file shows "Training finished. RMSE Energy: ...".

### Scenario 04-02: "Active Set Selection" (Priority: Medium)
**Objective**: Verify that the system can select a subset of structures based on D-optimality.

1.  **Preparation**:
    *   Create a large pool of 100 random structures.
    *   Set `config.yaml` to select 10 active structures.
2.  **Action**: Run `ActiveSetSelector.select(pool, n=10)`.
3.  **Expectation**:
    *   The returned list has length 10.
    *   The selected structures are distinct from each other (high diversity).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Potential Training

  Scenario: Train with Delta Learning
    GIVEN a labelled dataset of atomic structures
    AND a configuration specifying "LJ Baseline"
    WHEN the Trainer executes the fitting process
    THEN the system should calculate LJ parameters for all element pairs
    AND configure Pacemaker to fit the residual energy (DFT - LJ)
    AND output a valid .yace potential file

  Scenario: Optimize Training Set
    GIVEN a large pool of candidate structures
    WHEN the Active Set Selector runs MaxVol algorithm
    THEN the system should identify the subset of structures with highest information content
    AND discard redundant structures to save computational cost
```
