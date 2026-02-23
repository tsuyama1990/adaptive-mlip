# Architectural Analysis

## 1. Current State (Review)
The current `pyacemaker` implementation provides a foundational structure for an active learning loop.
- **Domain Models**: Pydantic models in `src/pyacemaker/domain_models` define the configuration schema. They are strict and well-structured.
- **Core ABCs**: `src/pyacemaker/core/base.py` defines `BaseGenerator`, `BaseOracle`, `BaseTrainer`, and `BaseEngine`, using `Iterator[Atoms]` for streaming. This is a pragmatic design choice for scalability.
- **Orchestrator**: `src/pyacemaker/orchestrator.py` manages the workflow. It implements a simple linear loop (Generate -> Label -> Train -> Run) and uses a structured directory layout (`active_learning/iter_XXX/`).

## 2. Comparison with Spec (`ALL_SPEC.md`)
The implementation diverges from the spec in several key areas, largely reflecting a pragmatic "Cycle 01" implementation:
- **Directory Structure**: The implemented structure aligns well with the spec, organizing artifacts into iterations.
- **Workflow Complexity**: The current linear loop is simpler than the spec's "Halt & Diagnose" cycle. This is an appropriate simplification for the initial phase, with modular methods (`_explore`, `_label`, etc.) allowing future extension.
- **Configuration**: The configuration is simpler than the spec's adaptive policy requirements but remains strict.

## 3. Design Decisions & Refactoring
Based on the principle of "Pragmatic Design over Rigid Spec", the following decisions have been confirmed and implemented:

### A. Directory Structure (Maintained)
The structured directory layout in `Orchestrator` is maintained as it provides better organization than a flat directory.

### B. Configuration Refactoring (Improved Usability)
- **Pseudopotential Validation**: The initial implementation of `DFTConfig` enforced overly strict path validation (requiring files to be within the project directory). This has been refactored to allow absolute paths to existing files (e.g., system libraries) while maintaining security against path traversal via relative paths.
- **Pacemaker Configuration**: `PacemakerTrainer` has been hardened to ensure robust configuration generation and correct handling of external parameters.

### C. Type Safety (Enforced)
- **Policy Selection**: `StructureGenerator` now uses strict Enum typing for policy selection to prevent runtime errors.
- **Static Analysis**: The codebase is fully typed (`mypy` strict) and linted (`ruff`).

### D. Streaming & Scalability (Maintained)
The `Iterator[Atoms]` interface in the core ABCs is preserved to ensure O(1) memory usage, a critical requirement for scalability.
