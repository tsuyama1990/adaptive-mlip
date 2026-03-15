# PyAceMaker - Adaptive MLIP GUI Platform

## Overview
PyAceMaker is an advanced, automated framework that seamlessly connects Quantum ESPRESSO (DFT), LAMMPS (Molecular Dynamics), MACE (Machine Learning Interatomic Potentials), and EON (Transition State Search). It is specifically designed to completely abstract away the "CUI script language" paradigm. By translating simple high-level inputs ("Intent-Driven") into heavily optimized backend execution pipelines, PyAceMaker solves the massive technical debt and cognitive load experienced by materials scientists configuring On-The-Fly (OTF) Active Learning simulations.

## Key Features
* **Intent-Driven Semantic Compiler**: Automatically translates your goal (e.g. "Active Learning for Platinum with Accuracy Level 5") into rigorous Pydantic schemas and executable C++ commands (like LAMMPS data files).
* **Visual Spatial Tagging**: Select and define physical constraints directly on a 3D interface! Paint the bottom layers of a slab, tag them as `ACTION_FREEZE`, and PyAceMaker will use rigorous Numpy boolean masking to generate exact `region`, `group`, and `fix setforce` text constraints for LAMMPS dynamically.
* **On-The-Fly Active Learning**: MD simulations are monitored continuously. When uncertainty crosses a threshold, Quantum ESPRESSO automatically calculates exact forces and fine-tunes the Machine Learning Potential without user intervention.
* **Intelligent Edge Handling**: Overlapping user-drawn visual constraints are deterministically resolved (e.g. `FREEZE` overrides `THERMOSTAT`) safely before simulation begins.

## Installation
Ensure you have `uv` installed.
```bash
uv sync
```

## Usage
Run the FastAPI GUI Backend server:
```bash
uv run pyacemaker gui --port 8000
```
This boots up the `pyacemaker.main.app` endpoint waiting to receive visual payloads (e.g., `IntentRequest`) directly from the frontend interface!

For legacy simulations via YAML config:
```bash
uv run pyacemaker run --config config.yaml
```

## Directory Structure
* `src/pyacemaker/domain_models/` - Core Intent-driven Schema definitions including GUI mappings.
* `src/pyacemaker/api/` - FastAPI endpoints for the React frontend.
* `src/pyacemaker/utils/` - Mathematical masking and spatial logic engines.
* `src/pyacemaker/core/` - Generators for LAMMPS text execution.
