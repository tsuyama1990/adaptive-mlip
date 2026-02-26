# Architectural Analysis

I have compared the code with the spec and found:

- **Structure**: The modular architecture (Orchestrator, Core, Interfaces, Domain Models) aligns well with the Spec.
- **Discrepancies**:
    - `MDConfig` lacks support for Temperature/Pressure ramping and Monte Carlo (MC) swap settings, which are critical for the "Exploration" phase described in the Spec.
    - `StructureGenerator` and `Policy` implementations are good but `MDMicroBurstPolicy` is buggy and `NormalModePolicy` is a placeholder.
    - `LammpsScriptGenerator` misses implementations for `fix atom/swap` (MC) and ramping `fix npt`.
    - `Orchestrator` has a placeholder `_adapt_strategy` which should be at least partially implemented or cleaned up.

## Decision

- **Prioritize Code Design**: Keep the current modular structure.
- **Fix Implementation**: Update `MDConfig`, `LammpsScriptGenerator`, and `Policies` to support the missing Spec features (MC, Ramping, MicroBurst) as they are essential for the "Active Learning" capability.
