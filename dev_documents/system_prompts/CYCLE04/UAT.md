# CYCLE 04: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-04-01 (Priority: Critical)
**Title:** The Production Loop and Checkpointing Resilience
**Description:** Verify the complete, automated execution of the 4-Phase architecture. Crucially, verify that if the HPC environment forcefully kills the PyAceMaker python process (e.g., due to a job timeout), the StateManager allows the system to instantaneously resume from the last completed major sub-task (Phase 1, DFT call, or Training) without repeating hours of calculation.

### Scenario ID: UAT-04-02 (Priority: Medium)
**Title:** Constant Storage Footprint (Artifact Cleanup)
**Description:** Ensure that the system automatically deletes or compresses enormous wave function files (`.wfc`) and LAMMPS trajectory dumps (`.dump`) once their useful information has been extracted and committed to the atomic database (`.extxyz`).

## 2. Behavior Definitions (Gherkin)

### UAT-04-01: The Production Loop and Checkpointing Resilience

```gherkin
FEATURE: HPC Fault Tolerance
  As a materials computational scientist
  I want my active learning loop to survive HPC node crashes and job timeouts
  So that I never lose days of progress when a million-step simulation is unexpectedly terminated.

  SCENARIO: Killing and resuming the 4-Phase Orchestrator
    GIVEN a fully configured PyAceMaker workflow running the 4-Phase architecture
    AND the Orchestrator successfully completed Phase 1 (Zero-Shot) and began the Phase 3/4 loop
    AND the StateManager saved `state.json` indicating Phase 3 is active
    WHEN I forcefully send a `kill -9` signal to the main python process during a simulated `QEDriver` calculation
    AND I later relaunch `uv run pyacemaker --config config.yaml` in the same directory
    THEN the Orchestrator instantly parses `state.json`
    AND skips Phase 1 entirely
    AND immediately resumes the execution exactly where it left off (re-running the interrupted DFT calculation)
    AND successfully transitions to Phase 4 upon completion.
```

### UAT-04-02: Constant Storage Footprint (Artifact Cleanup)

```gherkin
FEATURE: Artifact Daemon
  As a system architect
  I want the framework to manage its own storage footprint
  So that users do not hit HPC quota limits during month-long learning cycles.

  SCENARIO: Cleaning up `.wfc` and `.dump` files
    GIVEN an active PyAceMaker process executing the Production Loop
    AND the `QEDriver` successfully completes a calculation, leaving a 2GB `.save/wfc` directory
    AND the `LAMMPSEngine` halts, leaving a 5GB `dump.lammps` file
    WHEN the `Orchestrator` receives the resulting `Atoms` objects with extracted metadata
    THEN the `cleanup_artifacts` utility is triggered automatically
    AND the `.save/wfc` and `dump.lammps` files are deleted (or gzipped if configured)
    AND the critical `.yaml` configurations and `.extxyz` databases remain fully intact
    AND the total working directory size remains roughly constant (O(1) storage complexity).
```