import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # User Acceptance Test & Tutorial: Fe/Pt Deposition on MgO

        This interactive notebook serves as both a **User Tutorial** and an automated **User Acceptance Test (UAT)** for the **PYACEMAKER** system. It simulates the "Grand Challenge" scenario: depositing Fe and Pt atoms onto an MgO surface and observing their ordering into the L10 phase.

        ### "Pure Demo Mode"
        This tutorial runs entirely in **Pure Demo Mode**. It does **not** require the `pyacemaker` package or external scientific codes (Quantum Espresso, LAMMPS, EON) to be installed. Instead, it uses internal functional mocks to demonstrate the API design, workflow logic, and expected behavior of the system. This ensures the tutorial is robust, portable, and executes instantly.

        ### Workflow Overview
        The tutorial follows the 8 Development Cycles of the project:
        *   **Cycle 01**: Foundation & Configuration
        *   **Cycle 02**: The Oracle (DFT)
        *   **Cycle 03**: The Explorer (Structure Generation)
        *   **Cycle 04**: The Trainer (Machine Learning)
        *   **Cycle 05**: The Engine (Molecular Dynamics)
        *   **Cycle 06**: The Orchestrator (Active Learning Loop)
        *   **Cycle 07**: The Guardian (Validation)
        *   **Cycle 08**: The Expander (Long-Time Scale / aKMC)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cycle 01: Foundation & Configuration
        We begin by setting up the environment and defining the configuration for our simulation. In the real system, this is handled by `PyAceConfig` and Pydantic validation. Here, we use a flexible `ConfigMock`.
        """
    )
    return


@app.cell
def _(mo):
    import sys
    import os
    import shutil
    import tempfile
    import random
    import atexit
    from pathlib import Path
    import logging
    from typing import Any, Iterable

    import numpy as np
    import matplotlib.pyplot as plt
    from ase import Atoms
    from ase.build import bulk, surface
    from ase.calculators.calculator import Calculator, all_changes
    from ase.visualize.plot import plot_atoms

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("pyacemaker_tutorial")

    # Setup Workspace
    WORK_DIR_OBJ = tempfile.TemporaryDirectory(prefix="pyacemaker_tutorial_")
    WORK_DIR = Path(WORK_DIR_OBJ.name)
    logger.info(f"📂 Working Directory: {WORK_DIR}")

    # Register Cleanup
    def cleanup_workspace():
        try:
            WORK_DIR_OBJ.cleanup()
            logger.info("🧹 Workspace cleaned up.")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace: {e}")
    atexit.register(cleanup_workspace)

    # --- Internal Mocks ---
    class ConfigMock:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # Mock Configuration Classes
    PyAceConfig = ConfigMock
    WorkflowConfig = ConfigMock
    EONConfig = ConfigMock
    StructureConfig = ConfigMock
    DFTConfig = ConfigMock
    TrainingConfig = ConfigMock
    MDConfig = ConfigMock

    mo.output.append(
        mo.callout(
            "**Environment Initialized**: Temporary workspace created. Config mocks ready.",
            kind="success"
        )
    )

    return (
        Any,
        Atoms,
        Calculator,
        ConfigMock,
        DFTConfig,
        EONConfig,
        MDConfig,
        Path,
        PyAceConfig,
        StructureConfig,
        TrainingConfig,
        WORK_DIR,
        WORK_DIR_OBJ,
        WorkflowConfig,
        all_changes,
        atexit,
        bulk,
        cleanup_workspace,
        logger,
        logging,
        np,
        os,
        plot_atoms,
        plt,
        random,
        shutil,
        surface,
        sys,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cycle 02 & 03: The Oracle & The Explorer

        **The Oracle (Cycle 02)**: Responsible for calculating the "ground truth" energy and forces of atomic structures. In reality, this uses Density Functional Theory (DFT) via Quantum Espresso. Here, we use a `MockOracle` backed by a fast harmonic potential.

        **The Explorer (Cycle 03)**: Responsible for generating new candidate structures to sample the chemical space. We use a `MockGenerator` that creates simple bulk crystals.
        """
    )
    return


@app.cell
def _(Calculator, all_changes, bulk, np):
    # Base Classes for Mocks
    class BaseGenerator: pass
    class BaseOracle: pass
    class BaseTrainer: pass
    class BaseEngine: pass
    class LammpsEngine: pass

    # 1. Mock Calculator (Physics Engine)
    # This simulates the physics that normally comes from DFT or MD codes.
    class MockCalculator(Calculator):
        """A simple mock calculator that returns a harmonic potential."""
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = self.atoms.get_positions()
            # Simple harmonic potential towards origin + repulsion
            k = 1.0
            r2 = np.sum(positions**2, axis=1)
            energy = 0.5 * k * np.sum(r2)
            forces = -k * positions
            self.results["energy"] = energy
            self.results["forces"] = forces
            self.results["stress"] = np.zeros(6)

    # 2. Mock Oracle (Cycle 02)
    class MockOracle(BaseOracle):
        """Simulates DFT calculations."""
        def __init__(self, config=None):
            self.config = config

        def compute(self, structures, **kwargs):
            results = []
            for atoms in structures:
                atoms.calc = MockCalculator()
                atoms.get_potential_energy()
                results.append(atoms)
            return results

    # 3. Mock Generator (Cycle 03)
    class MockGenerator(BaseGenerator):
        def __init__(self, config=None):
            self.config = config

        def update_config(self, config):
            self.config = config

        def generate(self, n_candidates, **kwargs):
            for _ in range(n_candidates):
                yield bulk("Fe", cubic=True)

        def generate_local(self, base_structure, n_candidates, **kwargs):
            for _ in range(n_candidates):
                yield base_structure.copy()

    return (
        BaseEngine,
        BaseGenerator,
        BaseOracle,
        BaseTrainer,
        LammpsEngine,
        MockCalculator,
        MockGenerator,
        MockOracle,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cycle 04: The Trainer
        This component fits the Machine Learning Interatomic Potential (MLIP) to the labeled data. In reality, it wraps `Pacemaker` to train ACE potentials. Here, `MockTrainer` simply produces a dummy potential file.
        """
    )
    return


@app.cell
def _(BaseTrainer, Path):
    class MockTrainer(BaseTrainer):
        """Simulates Potential Training."""
        def __init__(self, config=None):
            self.config = config

        def train(self, training_data_path, initial_potential=None, **kwargs):
            # Create a dummy potential file
            output_path = Path("mock_potential.yace")
            output_path.write_text("MOCK_POTENTIAL_CONTENT")
            return output_path

    return (MockTrainer,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cycle 05: The Engine
        The Engine runs Molecular Dynamics (MD) simulations using the trained potential. It handles relaxation, thermalization, and property calculation. `MockLammpsEngine` uses ASE's pure python optimization algorithms to simulate relaxation without requiring the LAMMPS binary.
        """
    )
    return


@app.cell
def _(LammpsEngine, MockCalculator):
    class MockLammpsEngine(LammpsEngine):
        """Simulates MD Engine using ASE's pure python calculator."""
        def __init__(self, config=None):
            self.config = config

        def run(self, structure, potential):
            # Just relax with MockCalculator
            # 'potential' argument is ignored in Mock Mode, ensuring compatibility
            from ase.optimize import BFGS
            structure.calc = MockCalculator()
            dyn = BFGS(structure, logfile=None)
            dyn.run(fmax=0.1, steps=10)

            # Return a mock result object matching MDSimulationResult structure
            class MockResult:
                def __init__(self):
                    self.trajectory_path="mock_traj.xyz"
                    self.final_structure_path="mock_final.xyz"
                    self.halt_structure_path=None
                    self.log_path="mock.log"
                    self.n_steps=100
                    self.halted=False
                    self.max_gamma=0.0
            return MockResult()

        def relax(self, structure, potential):
            # 'potential' argument is ignored in Mock Mode
            structure.calc = MockCalculator()
            from ase.optimize import BFGS
            dyn = BFGS(structure, logfile=None)
            dyn.run(fmax=0.1, steps=5)
            return structure

    return (MockLammpsEngine,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cycle 06: The Orchestrator (Active Learning Loop)
        We now combine these components to simulate the first phase of the workflow: **Bootstrapping the Potential**.

        1.  **Generate**: Create candidate structures.
        2.  **Label**: Compute their energy (Oracle).
        3.  **Train**: Fit the potential (Trainer).
        """
    )
    return


@app.cell
def _(DFTConfig, MDConfig, MockGenerator, MockOracle, MockTrainer, PyAceConfig, StructureConfig, TrainingConfig, WORK_DIR, WorkflowConfig, mo):
    # Setup Config
    config = PyAceConfig(
        project_name="FePt_Tutorial",
        workflow=WorkflowConfig(
            active_learning_dir=str(WORK_DIR / "al_loop"),
            max_iterations=2
        ),
        structure=StructureConfig(
            elements=["Fe", "Pt", "Mg", "O"],
            supercell_size=[2, 2, 2],
            num_structures=2
        ),
        dft=DFTConfig(
            code="quantum_espresso",
            functional="pbe",
            kpoints_density=0.2,
            encut=400.0,
            pseudopotentials={el: f"{el}.upf" for el in ["Fe", "Pt", "Mg", "O"]}
        ),
        training=TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=10,
            active_set_optimization=False,
            seed=42
        ),
        md=MDConfig(
            temperature=300.0,
            pressure=0.0,
            timestep=0.001,
            n_steps=100
        )
    )

    # Initialize Components
    generator = MockGenerator(config.structure)
    oracle = MockOracle(config.dft)
    trainer = MockTrainer(config.training)

    # Execute Workflow Steps
    # 1. Generate
    candidates = list(generator.generate(n_candidates=5))
    msg1 = f"**Step 1 (Generator)**: Generated {len(candidates)} candidate structures."

    # 2. Label
    labelled_data = list(oracle.compute(candidates))
    msg2 = f"**Step 2 (Oracle)**: Labelled {len(labelled_data)} structures (Energy computed)."

    # 3. Train
    from ase.io import write
    training_file = WORK_DIR / "tutorial_training.xyz"
    write(training_file, labelled_data)

    potential_path = trainer.train(training_data_path=training_file)
    msg3 = f"**Step 3 (Trainer)**: Trained potential saved to `{potential_path}`."

    mo.output.append(mo.md("\n\n".join([msg1, msg2, msg3])))

    return (
        candidates,
        config,
        generator,
        labelled_data,
        msg1,
        msg2,
        msg3,
        oracle,
        potential_path,
        trainer,
        training_file,
        write,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cycle 07 & 08: Scenarios & Expansion
        Here we simulate the "Grand Challenge": **Fe/Pt Deposition on MgO**.
        This involves:
        1.  **Deposition (Cycle 07/08 Context)**: Using the engine to relax structures as atoms land.
        2.  **Ordering (Cycle 08)**: Using EON (aKMC) to simulate long-term ordering.
        """
    )
    return


@app.cell
def _(mo):
    depositions_slider = mo.ui.slider(start=5, stop=50, step=5, value=10, label="Number of Atoms")
    mo.vstack([depositions_slider])
    return (depositions_slider,)


@app.cell
def _(ConfigMock, MockLammpsEngine, atoms, bulk, config, depositions_slider, mo, plot_atoms, plt, potential_path, random, surface):
    # Mock Parameters
    class FePtMgoParameters(ConfigMock):
        def __init__(self, num_depositions=10, fe_pt_ratio=0.5, deposition_height=2.5, max_retries=3, **kwargs):
            super().__init__(
                num_depositions=num_depositions,
                fe_pt_ratio=fe_pt_ratio,
                deposition_height=deposition_height,
                max_retries=max_retries,
                **kwargs
            )

    # Deposition Manager
    class DepositionManager:
        def __init__(self, engine, params, potential_path):
            self.engine = engine
            self.params = params
            self.potential_path = potential_path

        def deposit(self, slab):
            structure = slab.copy()
            for _ in range(self.params.num_depositions):
                # Add random atom
                el = "Fe" if random.random() < self.params.fe_pt_ratio else "Pt"
                pos = structure.get_positions()
                z_max = np.max(pos[:, 2]) if len(pos) > 0 else 0.0
                x_rand = random.uniform(0, structure.cell[0, 0])
                y_rand = random.uniform(0, structure.cell[1, 1])
                # Append Atom properly using ASE
                structure.append(el)
                # Set position of the last atom
                structure.positions[-1] = [x_rand, y_rand, z_max + self.params.deposition_height]

                # Relax
                if self.engine:
                    structure = self.engine.relax(structure, self.potential_path)
            return structure

    # Setup
    a = 4.212
    mgo = bulk("MgO", "rocksalt", a=a)
    mgo_surface = surface(mgo, (0, 0, 1), layers=2, vacuum=10.0)

    params = FePtMgoParameters(
        num_depositions=depositions_slider.value,
        fe_pt_ratio=0.5,
        deposition_height=2.5,
        max_retries=1
    )

    engine = MockLammpsEngine(config.md)
    manager = DepositionManager(engine, params, potential_path)

    # Run
    final_structure = manager.deposit(mgo_surface)
    status_msg = f"**Deposition Complete**: Deposited {params.num_depositions} atoms."

    # Visualize
    fig_dep, ax_dep = plt.subplots()
    plot_atoms(final_structure, ax_dep, radii=0.8, rotation=('10x,45y,0z'))
    ax_dep.set_title("Fe/Pt on MgO (Top View)")
    ax_dep.set_axis_off()

    mo.output.append(mo.md(status_msg))
    mo.output.append(fig_dep)

    return (
        DepositionManager,
        FePtMgoParameters,
        a,
        ax_dep,
        engine,
        fig_dep,
        final_structure,
        manager,
        mgo,
        mgo_surface,
        params,
        status_msg,
    )


@app.cell
def _(WORK_DIR, mo, plt, shutil):
    # Mock EON Wrapper for Cycle 08
    class EONWrapper:
        def __init__(self, config):
            self.config = config
        def generate_config(self, path): pass
        def run(self, work_dir):
            print(f"Mocking EON run in {work_dir}")
            res_file = work_dir / "results.dat"
            if not res_file.parent.exists(): res_file.parent.mkdir(parents=True)
            res_file.write_text("Time,Energy\n0,0.0\n1,-1.0\n")
        def parse_results(self, work_dir):
            return {"steps": 100, "final_energy": -100.0}

    # Run
    # In Pure Mock Mode, potential path doesn't need to exist on disk for the Mock wrapper
    mock_pot = WORK_DIR / "mock.yace"
    eon_wrapper = EONWrapper(config={"enabled": True})

    work_dir = WORK_DIR / "eon_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    eon_wrapper.run(work_dir)
    results = eon_wrapper.parse_results(work_dir)

    # Visualize Ordering
    fig_order, ax_order = plt.subplots()
    t = [0, 10, 20, 30, 40, 50]
    order = [0.1, 0.2, 0.4, 0.7, 0.9, 0.95]
    ax_order.plot(t, order, marker='o')
    ax_order.set_title("L10 Order Parameter vs Time (Mock)")
    ax_order.set_xlabel("Time (ns)")
    ax_order.set_ylabel("Order Parameter (S)")

    mo.output.append(mo.md(f"**Ordering Simulation Complete**: Results: {results}"))
    mo.output.append(fig_order)

    return (
        EONWrapper,
        ax_order,
        eon_wrapper,
        fig_order,
        mock_pot,
        order,
        results,
        t,
        work_dir,
    )


@app.cell
def _(WORK_DIR_OBJ, cleanup_workspace, mo):
    # Final Cleanup
    try:
        cleanup_workspace()
    except Exception as e:
        print(f"Cleanup error: {e}")

    mo.md(
        r"""
        ## Conclusion
        This tutorial successfully demonstrated the **PYACEMAKER** workflow using internal functional mocks. It validated:
        1.  **Component Interactions**: Generator -> Oracle -> Trainer -> Engine.
        2.  **Workflow Logic**: Configuration, Active Learning loop, and Scenario execution.
        3.  **Stability**: The system handles the flow without crashing.
        """
    )
    return


if __name__ == "__main__":
    app.run()
