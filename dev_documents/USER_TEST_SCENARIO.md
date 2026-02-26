# User Acceptance Testing (UAT) & Tutorial Master Plan

## 1. Overview
This document outlines the strategy for User Acceptance Testing (UAT) and the creation of executable tutorials for the **PyAceMaker** system. The primary goal is to verify the efficacy of the **MACE Knowledge Distillation & Delta Learning** workflow (Cycles 01-06).

## 2. Tutorial Strategy

We will adopt a dual-mode execution strategy to ensure that the tutorials serve both as educational material for users and as rigorous integration tests for the development team.

### 2.1. Mock Mode (CI/Quick Start)
-   **Objective**: Verify the software logic and pipeline orchestration without requiring heavy computational resources or external API keys.
-   **Method**:
    -   DFT calculations are replaced by a "Mock Oracle" (e.g., a simple Lennard-Jones potential).
    -   MACE model loading is simulated or uses a tiny random model.
    -   Training steps are shortened (1 epoch).
-   **Trigger**: Activated when the environment variable `CI=true` or `PYACEMAKER_MODE=MOCK` is set.
-   **Expected Runtime**: < 5 minutes.

### 2.2. Real Mode (Scientific Validation)
-   **Objective**: Reproduce actual scientific results and demonstrate the system's accuracy.
-   **Method**:
    -   Uses real DFT codes (VASP/Quantum Espresso) or a high-fidelity surrogate (MACE-MP-0).
    -   Performs full training and fine-tuning.
-   **Trigger**: Activated by default or when `PYACEMAKER_MODE=REAL`.
-   **Expected Runtime**: Hours to Days (depending on hardware).

## 3. Tutorial Plan: `tutorials/UAT_AND_TUTORIAL.py`

A **SINGLE** executable file named `tutorials/UAT_AND_TUTORIAL.py` (formatted as a Marimo notebook) will be created. This file will contain the complete end-to-end workflow for the SN2 Reaction scenario described below.

### 3.1. File Structure
The script will be structured into the following sections:
1.  **Setup & Configuration**: Importing libraries, setting up the `config.yaml`, and detecting the execution mode (Mock/Real).
2.  **Step 1: DIRECT Sampling**: Visualizing the generated initial structures.
3.  **Step 2: Active Learning**: Visualizing the selection of high-uncertainty structures.
4.  **Step 3 & 4: Surrogate Loop**: Displaying the MACE fine-tuning loss and MD trajectories.
5.  **Step 5 & 6: Base Training**: Labeling the surrogate data and showing the initial ACE potential's performance.
6.  **Step 7: Delta Learning**: Demonstrating the accuracy improvement after fine-tuning with DFT data.
7.  **Analysis**: Plotting the Energy Barrier (NEB) comparison between DFT, MACE, and the final ACE potential.

## 4. Test Scenario: SN2 Reaction (Transition State)

**Target System**: Methyl Chloride SN2 Reaction ($CH_3Cl + OH^- \rightarrow CH_3OH + Cl^-$)
**Goal**: Accurately reproduce the energy barrier of the transition state (TS) using the generated ACE potential.

### 4.1. Prerequisites
-   Initial structures for Reactant and Product.
-   Access to VASP/QE (for Real Mode) or Mock Oracle (for CI).

### 4.2. Step-by-Step Execution
1.  **Initialization**: The system initializes with `elements: ["C", "H", "O", "Cl"]`.
2.  **DIRECT Sampling**: Generates 50 diverse configurations around the reaction path.
3.  **Active Learning**: Selects the top 10 most uncertain structures (likely near the TS) for DFT labeling.
4.  **MACE Fine-tuning**: Adapts MACE-MP-0 to this specific reaction.
5.  **Surrogate Sampling**: Runs high-temperature MD to sample the phase space broadly.
6.  **Labeling & Training**: Creates a dataset of 500+ structures and trains the base ACE model.
7.  **Delta Learning**: Fine-tunes the ACE potential using the high-accuracy DFT data from Step 3.

### 4.3. Validation Criteria
-   **Mock Mode**: The pipeline completes all 7 steps without error. Artifacts (`final_potential.yace`) are created.
-   **Real Mode**: The calculated activation energy ($E_a$) matches the DFT reference within 0.05 eV.

## 5. Validation Instructions

To validate the tutorial and system:

```bash
# 1. Install dependencies
uv sync

# 2. Run in Mock Mode (Fast Verification)
export PYACEMAKER_MODE=MOCK
python tutorials/UAT_AND_TUTORIAL.py
# OR if using marimo
marimo run tutorials/UAT_AND_TUTORIAL.py

# 3. Check Outputs
ls -l uat_sn2_reaction/final_potential.yace
```
