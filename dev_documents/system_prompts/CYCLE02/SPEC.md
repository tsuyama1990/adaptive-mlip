# CYCLE 02: Master-Slave Inversion & Seamless Resume

## 1. Summary

This cycle fundamentally alters how PYACEMAKER interacts with Molecular Dynamics (MD) engines. The goal is to solve the critical "MD Continuity Break" and "System Fragility" issues identified in the PRD. We will transition from a traditional "Python drives LAMMPS" model to a "Master-Slave Inversion" where LAMMPS is the primary driver, executing Python logic via callbacks (`fix python/invoke`), or alternatively, using robust `read_restart` mechanisms to preserve exact microscopic states (coordinates, velocities, thermostat states) when the potential needs updating. This cycle ensures that when uncertainty spikes, the simulation pauses, updates its knowledge base, and resumes seamlessly without losing the time-evolution of the physical system.

## 2. System Architecture

This cycle focuses entirely on the MD execution environment and its interaction with the python orchestration layer.

```text
src/pyacemaker/
├── core/
│   ├── base.py                 (No changes)
│   ├── **engine.py**           (Modify: Seamless Resume & Soft Start)
│   └── loop.py                 (No changes)
├── domain_models/
│   ├── config.py               (No changes from Cycle 01)
│   └── data.py
└── interfaces/
    └── **lammps_driver.py**    (Modify: Implement callback/restart logic)
```

## 3. Design Architecture

The core challenge is maintaining the state of a highly nonlinear dynamical system (MD) across potential updates.

### 3.1. Engine Module (`core/engine.py` & `interfaces/lammps_driver.py`)

We will heavily modify the `LAMMPSEngine` (or equivalent driver) to support the new paradigm.

**Key Concepts & Invariants:**

*   **State Preservation (The "Resume" Contract):** An MD run must be able to halt at step $N$, and upon restart, step $N+1$ must have exactly the positions, velocities, and ensemble properties (e.g., Nose-Hoover chain states) as if it had never stopped.
*   **The Soft Start Protocol:** When a potential is updated via Delta Learning, the energy landscape changes slightly. Resuming MD instantaneously with the old velocities on the new landscape can cause massive force spikes ("exploding" atoms). A "Soft Start" is mandatory: upon resume, the system must undergo $M$ steps of strong Langevin thermalization (`fix langevin`) to absorb the energy shock before returning to the production ensemble (e.g., NPT/NVT).
*   **Two-Tier Threshold Integration:** The engine must query the model's uncertainty at regular intervals. It must implement the `smooth_steps` logic defined in `ActiveLearningThresholds`: a single spike above `threshold_call_dft` is ignored; it must persist for `smooth_steps` to trigger a halt.

**Implementation Strategy (Callback vs. Restart):**

1.  **Primary Approach: `fix python/invoke` (In-memory Inversion):** LAMMPS runs a continuous loop. Every $K$ steps, it calls a Python function evaluating uncertainty. If high, Python pauses LAMMPS, triggers the Orchestrator (Phase 3 & 4), updates the `pair_coeff`, and tells LAMMPS to continue. (Fastest, but technically complex due to C++ memory management).
2.  **Fallback Approach: `write_restart` / `read_restart` (Process Isolation):** Python runs LAMMPS as a subprocess. LAMMPS writes a `.restart` file every $K$ steps. A separate Python watchdog monitors a dump file for uncertainty. If it spikes, Python kills the LAMMPS process, runs the update, and launches a new LAMMPS process using `read_restart` from the last safe point. (Slower, but extremely robust against LAMMPS crashes).

*Note: The implementation will default to the robust **Fallback Approach (Restart)** to guarantee stability, with the architectural hooks ready for the Primary Approach if performance dictates.*

## 4. Implementation Approach

1.  **Refactor `LAMMPSEngine.run()`:**
    *   Modify the main execution loop to accept an optional `restart_file` argument.
    *   If provided, the generated LAMMPS script must start with `read_restart {restart_file}` instead of initializing velocities and boxes.
2.  **Implement the Uncertainty Watchdog (Two-Tier Evaluator):**
    *   Create a method `_evaluate_uncertainty_stream(dump_file, thresholds)`.
    *   This function streams the LAMMPS dump output (containing per-atom $\gamma$ values if using PACE, or coordinates to be evaluated by MACE).
    *   Maintain a rolling window of length `smooth_steps`. If the system max uncertainty exceeds `threshold_call_dft` for the entire window, return `HALT` and the specific step number.
    *   Identify the "epicenter" atoms (those exceeding `threshold_add_train`) and return their indices.
3.  **Implement the Soft Start Generator:**
    *   Create a helper method `_generate_soft_start_commands(temperature, steps)`.
    *   This generates LAMMPS commands to apply a `fix langevin` with a very short damping parameter (e.g., 0.1 ps) for a specified number of steps (e.g., 100), followed by `unfix langevin` and restoration of the original thermostat.
4.  **Integrate Checkpointing:**
    *   Ensure the LAMMPS script always includes `restart {K} {prefix}.restart` commands.
5.  **Refine & Lint:** Run Ruff and MyPy. Ensure subprocess management uses robust `try/finally` blocks to prevent zombie processes.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)

Unit testing will focus on the logic of script generation, restart file handling, and the critical two-tier threshold smoothing algorithm.

1.  **LAMMPS Script Generation (Restart Mode):**
    *   Call the `LAMMPSEngine` script generator with a dummy `restart_file` path.
    *   **Assertions:** Verify the resulting script string begins with `read_restart` and correctly omits standard initialization commands (`create_box`, `velocity create`). Verify the new potential definition (`pair_style pace`, `pair_coeff`) is injected *after* reading the restart.
2.  **Soft Start Command Generation:**
    *   Generate a script with soft-start enabled.
    *   **Assertions:** Verify `fix langevin` is present, followed by a `run` command for the soft-start duration, an `unfix langevin`, and the setup of the primary production thermostat (e.g., `fix npt`).
3.  **Two-Tier Evaluator Logic (`_evaluate_uncertainty_stream`):**
    *   Create a mock stream of uncertainty values (a list of floats simulating steps).
    *   Test Case A (Thermal Noise): A single spike of `0.1` (threshold `0.05`) followed by `0.01`. The evaluator must NOT return `HALT`.
    *   Test Case B (Real Uncertainty): `0.06`, `0.07`, `0.08` for `smooth_steps=3`. The evaluator MUST return `HALT` at the 3rd step.
    *   **Assertions:** The function correctly filters noise and triggers only on sustained uncertainty, returning the correct halting step and the list of atom indices that exceeded `threshold_add_train`.
4.  **Process Isolation / Teardown:**
    *   Mock the `subprocess.Popen` call.
    *   Simulate a crash in LAMMPS (e.g., return code 1).
    *   **Assertions:** The Python engine must catch the exception, clean up any lingering temporary files, and raise a custom `MDHaltError` rather than completely crashing the Python interpreter.

### Integration Testing Approach (Min 300 words)

Integration testing will simulate the full "halt-and-resume" cycle using a fast empirical potential (like Lennard-Jones) within LAMMPS to verify seamless continuity.

1.  **The Seamless Resume Test:**
    *   Set up a small Argon system (LJ potential) using the `LAMMPSEngine`.
    *   Run a 100-step NVE simulation. This is the "Ground Truth". Record the total energy, positions, and velocities at step 100.
    *   Now, run the identical system but configure the `LAMMPSEngine` to artificially "halt" at step 50 and write a restart file.
    *   Instantiate a new `LAMMPSEngine` run, passing the step 50 restart file, and run for another 50 steps.
    *   **Assertions:** Compare the final state (step 100) of the interrupted run with the Ground Truth run. The positions, velocities, and total energy must match exactly (within numerical precision limits of the restart file format). This proves the "Resume" contract is mathematically sound.
2.  **Soft Start Stability Test:**
    *   Set up a system and run a short MD.
    *   Artificially perturb the potential parameters (e.g., increase the LJ epsilon by 10%) to simulate a Delta Learning update.
    *   Resume the simulation from a restart file *without* a soft start. (This should likely crash LAMMPS due to "Lost Atoms" or huge energy spikes, though we may just observe a massive temperature spike).
    *   Resume the simulation *with* the Soft Start protocol enabled.
    *   **Assertions:** Parse the LAMMPS thermodynamic output. The temperature must briefly fluctuate but remain bounded by the Langevin thermostat, stabilizing at the target temperature without crashing. The simulation must successfully complete the requested steps.
