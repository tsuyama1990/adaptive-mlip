# Architectural Analysis

## 1. Current State (Review)
The current `pyacemaker` implementation provides a foundational structure for an active learning loop.
- **Domain Models**: Pydantic models in `src/pyacemaker/domain_models` define the configuration schema. They are strict and well-structured.
- **Core ABCs**: `src/pyacemaker/core/base.py` defines `BaseGenerator`, `BaseOracle`, `BaseTrainer`, and `BaseEngine`, using `Iterator[Atoms]` for streaming. This is a pragmatic design choice for scalability.
- **Orchestrator**: `src/pyacemaker/orchestrator.py` manages the workflow. It implements a simple linear loop (Generate -> Label -> Train -> Run) and dumps artifacts into a single `data/` directory.

## 2. Comparison with Spec (`ALL_SPEC.md`)
The implementation diverges from the spec in several key areas:
- **Directory Structure**: The spec mandates a structured directory layout (`active_learning/iter_XXX/md_run`, `dft_calc`, etc.). The current code uses a flat `data/` directory.
- **Workflow Complexity**: The spec describes a complex "Explore -> Detect -> Select -> Refine -> Deploy" cycle with "Halt & Diagnose" logic. The current code is a linear loop without dynamic halting or complex selection.
- **Config Depth**: The spec implies a rich configuration for adaptive policies. The current config is minimal.

## 3. Design Decisions
Based on the principle of "Pragmatic Design over Rigid Spec", the following decisions have been made:

### A. Adopt Structured Directory Layout (Fix Implementation)
The flat `data/` directory is insufficient for debugging and provenance tracking in a real-world active learning campaign. We will adopt the spec's directory structure:
- `active_learning/iter_{n:03d}/`
  - `candidates/` (Generated structures)
  - `dft_calc/` (Oracle calculations)
  - `training/` (Training data and potentials)
  - `md_run/` (MD simulation artifacts)
- `potentials/` (Deployed potentials)

### B. Modularize Orchestrator (Refactor for Extensibility)
The current `_run_active_learning_step` is monolithic. We will break it down into distinct methods:
- `_explore()`: Handles Generator or MD exploration.
- `_label()`: Handles Oracle computations.
- `_train()`: Handles Potential training.
- `_deploy()`: Handles Potential deployment.

This prepares the codebase for the future "Halt & Diagnose" logic without over-engineering it now.

### C. Config Evolution (Keep Simple but Strict)
We will extend `WorkflowConfig` to support the new directory structure fields (`active_learning_dir`, `potentials_dir`) but avoid adding speculative fields for features not yet implemented (like adaptive policies).

### D. Streaming & Scalability (Maintain Pragmatic Choice)
We will strictly maintain the `Iterator[Atoms]` interface in the core ABCs, as this is superior to loading all structures into memory, aligning with the "Scalability" goal of the spec.
