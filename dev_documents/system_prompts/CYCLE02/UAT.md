# CYCLE 02: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-02-01 (Priority: Critical)
**Title:** Seamless MD Resume
**Description:** Verify that the system can halt a running Molecular Dynamics simulation mid-trajectory due to high uncertainty, perform an external operation (simulating a potential update), and seamlessly resume the simulation without any unphysical "jumps" in coordinates, velocities, or thermostat state.

### Scenario ID: UAT-02-02 (Priority: High)
**Title:** Thermal Noise Exclusion
**Description:** Ensure the two-tier threshold system correctly filters out single-step spikes in uncertainty (representing harmless thermal vibrations) and only halts the MD simulation when uncertainty is sustained over multiple steps.

## 2. Behavior Definitions (Gherkin)

### UAT-02-01: Seamless MD Resume

```gherkin
FEATURE: Master-Slave Inversion and Resume
  As a materials computational scientist
  I want my MD simulation to seamlessly resume after a potential update
  So that I do not lose critical time-evolution information (like diffusion pathways) across active learning cycles.

  SCENARIO: Halting and resuming a bulk LAMMPS simulation
    GIVEN a LAMMPS engine running an NPT ensemble on a 1000-atom supercell
    AND the `LAMMPSEngine` is configured to write `restart` files every 10 steps
    AND the engine is running step 0 to 100 with a dummy `base.yace`
    WHEN an external process forcefully halts the engine at step 50
    AND I invoke `LAMMPSEngine.run()` again, providing the step 50 `restart_file`
    THEN the simulation resumes from step 50 and completes step 100
    AND the final coordinates and velocities at step 100 are numerically identical to a continuous, un-interrupted run
    AND the Nose-Hoover chain thermostat remains perfectly contiguous.
```

### UAT-02-02: Thermal Noise Exclusion

```gherkin
FEATURE: Uncertainty Smoothing
  As a system architect
  I want the system to ignore harmless thermal vibrations
  So that I don't trigger expensive DFT calculations unnecessarily.

  SCENARIO: Filtering out instantaneous uncertainty spikes
    GIVEN an `ActiveLearningThresholds` configuration with `threshold_call_dft = 0.05`
    AND `smooth_steps = 3`
    AND a continuous stream of MD configurations being evaluated
    WHEN the uncertainty is `[0.01, 0.02, 0.06, 0.01, 0.02]` (single spike)
    THEN the `UncertaintyWatchdog` MUST NOT trigger a `HALT`
    WHEN the uncertainty is `[0.01, 0.06, 0.07, 0.08, 0.01]` (sustained spike)
    THEN the `UncertaintyWatchdog` MUST trigger a `HALT` exactly at the 3rd consecutive step > 0.05
    AND it returns the indices of atoms whose individual uncertainty > `threshold_add_train`.
```