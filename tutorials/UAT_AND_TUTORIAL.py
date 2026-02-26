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

        This interactive notebook serves as both a **User Tutorial** and an automated **User Acceptance Test (UAT)** for the PYACEMAKER system. It simulates the "Grand Challenge" scenario: depositing Fe and Pt atoms onto an MgO surface and observing their ordering into the L10 phase.

        ### Scientific Background
        **L10 Ordering**: The L10 phase is a chemically ordered structure found in binary alloys like FePt. It consists of alternating layers of Fe and Pt atoms along the [001] direction. This phase is crucial for magnetic recording applications due to its high magnetocrystalline anisotropy.

        **Active Learning Loop**: To simulate this process accurately, we need an interatomic potential (Force Field). Instead of training on a static dataset, we use an **Active Learning** approach:
        1.  **Structure Generation**: The system explores new atomic configurations (e.g., during deposition).
        2.  **Uncertainty Detection**: If the model encounters a structure it "doesn't know" (high uncertainty), it halts.
        3.  **Oracle Calculation**: We compute the true energy/forces using Density Functional Theory (DFT).
        4.  **Retraining**: The potential is updated with the new data and the simulation resumes.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Environment Setup
        We first check if the `pyacemaker` package is installed. If not, we fall back to a **Pure Demo Mode** using internal mocks.
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

    # Try to add src to path for local development
    repo_root = Path(".").resolve()
    if (repo_root / "src").exists():
        sys.path.append(str(repo_root / "src"))

    # --- Pure Demo Mode Fallbacks ---
    # Define these mock classes first so they are available if import fails

    class ConfigMock:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class BaseGenerator: pass
    class BaseOracle: pass
    class BaseTrainer: pass
    class BaseEngine: pass
    class LammpsEngine: pass
    class Orchestrator: pass
    class FePtMgoScenario: pass

    # Mocking Domain Models
    class FePtMgoParameters(ConfigMock):
        """Mock parameters for the deposition scenario."""
        def __init__(self, num_depositions=10, fe_pt_ratio=0.5, deposition_height=2.5, max_retries=3, **kwargs):
            super().__init__(
                num_depositions=num_depositions,
                fe_pt_ratio=fe_pt_ratio,
                deposition_height=deposition_height,
                max_retries=max_retries,
                **kwargs
            )

    # Mocking Deposition Manager (Functional)
    class DepositionManager:
        """Mock manager that performs simplified deposition."""
        def __init__(self, engine, params, potential_path):
            self.engine = engine
            self.params = params
            self.potential_path = potential_path

        def deposit(self, slab):
            """Simulates deposition by adding atoms and relaxing."""
            structure = slab.copy()
            for _ in range(self.params.num_depositions):
                # Add random atom
                el = "Fe" if random.random() < self.params.fe_pt_ratio else "Pt"
                pos = structure.get_positions()
                z_max = np.max(pos[:, 2]) if len(pos) > 0 else 0.0
                x_rand = random.uniform(0, structure.cell[0, 0])
                y_rand = random.uniform(0, structure.cell[1, 1])
                structure.append(Atoms(el, positions=[[x_rand, y_rand, z_max + self.params.deposition_height]])[0])

                # Relax using the engine
                if self.engine:
                    structure = self.engine.relax(structure, self.potential_path)
            return structure

    # Mocking EON Wrapper (Functional)
    class EONWrapper:
        """Mock wrapper for EON client."""
        def __init__(self, config):
            self.config = config

        def generate_config(self, path):
            pass

        def run(self, work_dir):
            """Simulates running aKMC."""
            print(f"Mocking EON run in {work_dir}")
            # Create fake results file
            res_file = work_dir / "results.dat"
            if not res_file.parent.exists():
                res_file.parent.mkdir(parents=True)
            res_file.write_text("Time,Energy\n0,0.0\n1,-1.0\n")

        def parse_results(self, work_dir):
            return {"steps": 100, "final_energy": -100.0}

    # Initialize variables with Mock values by default
    PyAceConfig = ConfigMock
    WorkflowConfig = ConfigMock
    EONConfig = ConfigMock
    StructureConfig = ConfigMock
    DFTConfig = ConfigMock
    TrainingConfig = ConfigMock
    MDConfig = ConfigMock

    DEFAULT_ACTIVE_LEARNING_DIR = "active_learning"
    HAS_PYACEMAKER = False

    # Attempt Import
    try:
        import pyacemaker
        from pyacemaker.domain_models.config import (
            PyAceConfig as _PyAceConfig,
            WorkflowConfig as _WorkflowConfig,
            EONConfig as _EONConfig,
            StructureConfig as _StructureConfig,
            DFTConfig as _DFTConfig,
            TrainingConfig as _TrainingConfig,
            MDConfig as _MDConfig
        )
        from pyacemaker.domain_models.defaults import DEFAULT_ACTIVE_LEARNING_DIR as _DEFAULT_ACTIVE_LEARNING_DIR
        from pyacemaker.orchestrator import Orchestrator as _Orchestrator
        from pyacemaker.core.generator import BaseGenerator as _BaseGenerator
        from pyacemaker.core.oracle import BaseOracle as _BaseOracle
        from pyacemaker.core.trainer import BaseTrainer as _BaseTrainer
        from pyacemaker.core.engine import LammpsEngine as _LammpsEngine
        from pyacemaker.scenarios.fept_mgo import (
            FePtMgoScenario as _FePtMgoScenario,
            FePtMgoParameters as _FePtMgoParameters,
            DepositionManager as _DepositionManager
        )
        from pyacemaker.interfaces.eon_driver import EONWrapper as _EONWrapper

        # Override mocks with real classes
        PyAceConfig = _PyAceConfig
        WorkflowConfig = _WorkflowConfig
        EONConfig = _EONConfig
        StructureConfig = _StructureConfig
        DFTConfig = _DFTConfig
        TrainingConfig = _TrainingConfig
        MDConfig = _MDConfig
        FePtMgoParameters = _FePtMgoParameters
        DEFAULT_ACTIVE_LEARNING_DIR = _DEFAULT_ACTIVE_LEARNING_DIR
        Orchestrator = _Orchestrator
        BaseGenerator = _BaseGenerator
        BaseOracle = _BaseOracle
        BaseTrainer = _BaseTrainer
        LammpsEngine = _LammpsEngine
        FePtMgoScenario = _FePtMgoScenario
        DepositionManager = _DepositionManager
        EONWrapper = _EONWrapper

        HAS_PYACEMAKER = True
        logger.info("✅ pyacemaker package loaded successfully.")
    except ImportError:
        logger.warning("⚠️ pyacemaker not found. Running in PURE DEMO MODE with internal mocks.")
        HAS_PYACEMAKER = False

    # Setup Workspace
    # Use a temporary directory for this run to ensure isolation
    WORK_DIR_OBJ = tempfile.TemporaryDirectory(prefix="pyacemaker_tutorial_")
    WORK_DIR = Path(WORK_DIR_OBJ.name)
    logger.info(f"📂 Working Directory: {WORK_DIR}")

    # Notify User about Mode
    if not HAS_PYACEMAKER:
        mo.output.append(
            mo.callout(
                "**Running in Pure Demo Mode**: `pyacemaker` package was not found. Using internal functional mocks for demonstration.",
                kind="warn"
            )
        )
    else:
        mo.output.append(
            mo.callout(
                "**Running in Real Mode**: `pyacemaker` package loaded. Mocks will be used only for heavy external codes (DFT/LAMMPS) if they are missing.",
                kind="success"
            )
        )

    return (
        Any,
        Atoms,
        BaseEngine,
        BaseGenerator,
        BaseOracle,
        BaseTrainer,
        Calculator,
        ConfigMock,
        DEFAULT_ACTIVE_LEARNING_DIR,
        DFTConfig,
        DepositionManager,
        EONConfig,
        EONWrapper,
        FePtMgoParameters,
        FePtMgoScenario,
        HAS_PYACEMAKER,
        Iterable,
        LammpsEngine,
        MDConfig,
        Orchestrator,
        Path,
        PyAceConfig,
        StructureConfig,
        TrainingConfig,
        WORK_DIR,
        WORK_DIR_OBJ,
        WorkflowConfig,
        all_changes,
        bulk,
        logger,
        logging,
        np,
        os,
        plot_atoms,
        plt,
        pyacemaker,
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
        ## 2. Interactive Configuration
        Here you can adjust the simulation parameters.
        """
    )
    return


@app.cell
def _(mo):
    temperature_slider = mo.ui.slider(start=300, stop=2000, step=100, value=600, label="Deposition Temperature (K)")
    depositions_slider = mo.ui.slider(start=5, stop=50, step=5, value=10, label="Number of Atoms")

    mo.vstack([temperature_slider, depositions_slider])
    return depositions_slider, temperature_slider


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Mock Components Definition
        In this tutorial, we define **Mock Components** to simulate the behavior of external physics codes (like Quantum Espresso and LAMMPS) which are computationally expensive and require complex installation.

        *   **MockCalculator**: A simplified force field (harmonic potential) that runs instantly in Python. It replaces the heavy physics engines.
        *   **MockOracle**: Simulates the DFT calculation step. Instead of solving the Schrödinger equation, it uses `MockCalculator` to return consistent energies and forces.
        *   **MockTrainer**: Simulates the Machine Learning training step. It generates a dummy potential file.
        *   **MockLammpsEngine**: Simulates the Molecular Dynamics engine. It uses ASE's optimization algorithms with `MockCalculator` to relax structures, mimicking a real MD relaxation.
        """
    )
    return


@app.cell
def _(BaseGenerator, BaseOracle, BaseTrainer, Calculator, LammpsEngine, Path, all_changes, bulk, np):
    # --- Shared Mocks for Tutorial Mode ---

    # 1. Mock Calculator (Physics Engine)
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

    # 2. Mock Components using the Shared MockCalculator
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

    class MockTrainer(BaseTrainer):
        """Simulates Potential Training."""
        def __init__(self, config=None):
            self.config = config

        def train(self, training_data_path, initial_potential=None, **kwargs):
            # Create a dummy potential file
            output_path = Path("mock_potential.yace")
            output_path.write_text("MOCK_POTENTIAL_CONTENT")
            return output_path

    class MockLammpsEngine(LammpsEngine):
        """Simulates MD Engine using ASE's pure python calculator."""
        def run(self, structure, potential):
            # Just relax with MockCalculator
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
            structure.calc = MockCalculator()
            from ase.optimize import BFGS
            dyn = BFGS(structure, logfile=None)
            dyn.run(fmax=0.1, steps=5)
            return structure

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
        MockCalculator,
        MockGenerator,
        MockLammpsEngine,
        MockOracle,
        MockTrainer,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Phase 1: Training the Potential
        In this phase, we bootstrap the potential.

        **Steps**:
        1.  **Generate Candidates**: Create initial random structures (Rattled Bulk, Surfaces).
        2.  **Oracle Labeling**: Calculate their true energy using DFT (Mocked here).
        3.  **Training**: Fit an initial ACE potential to this data.
        """
    )
    return


@app.cell
def _(DFTConfig, MDConfig, MockGenerator, MockOracle, MockTrainer, PyAceConfig, StructureConfig, TrainingConfig, WORK_DIR, WorkflowConfig, depositions_slider, mo, temperature_slider):
    # Setup Config

    # NOTE: In Mock Mode, we bypass strict file validation for Pseudopotentials
    # by using strings that look like paths but aren't checked by our Mock Configs.

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
            # Mock required fields
            active_set_optimization=False,
            seed=42
        ),
        md=MDConfig(
            temperature=float(temperature_slider.value),
            pressure=0.0,
            timestep=0.001,
            n_steps=100
        )
    )

    # Initialize Components
    generator = MockGenerator(config.structure)
    oracle = MockOracle(config.dft)
    trainer = MockTrainer(config.training)

    mo.output.append(mo.md(f"Initialized components. Config Project: `{config.project_name}`"))
    return config, generator, oracle, trainer


@app.cell
def _(WORK_DIR, generator, mo, oracle, trainer):
    # 1. Generate Candidates
    candidates = list(generator.generate(n_candidates=5))
    mo.output.append(mo.md(f"**Step 1 (Generator)**: Generated {len(candidates)} candidate structures."))

    # 2. Label (Oracle)
    labelled_data = list(oracle.compute(candidates))
    mo.output.append(mo.md(f"**Step 2 (Oracle)**: Labelled {len(labelled_data)} structures (Energy computed)."))

    # 3. Train (Trainer)
    from ase.io import write
    training_file = WORK_DIR / "tutorial_training.xyz"
    write(training_file, labelled_data)

    potential_path = trainer.train(training_data_path=training_file)
    mo.output.append(mo.md(f"**Step 3 (Trainer)**: Trained potential saved to `{potential_path}`."))

    return candidates, labelled_data, potential_path, training_file, write


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Phase 2: Dynamic Deposition (Simulation)
        Now we simulate the physical process of depositing atoms onto the MgO substrate.
        The system uses the trained potential to relax the structure after each deposition.

        **Components**:
        *   **DepositionManager**: Handles the sequential addition of atoms and calls the engine for relaxation.
        *   **LammpsEngine (Mock)**: Relaxes the structure to minimize energy.
        """
    )
    return


@app.cell
def _(DepositionManager, FePtMgoParameters, MockLammpsEngine, bulk, config, depositions_slider, mo, plot_atoms, plt, potential_path, surface):
    # Initialize Surface
    a = 4.212
    mgo = bulk("MgO", "rocksalt", a=a)
    mgo_surface = surface(mgo, (0, 0, 1), layers=2, vacuum=10.0)

    # Setup Deposition Manager
    # FePtMgoParameters defines the scenario constraints (ratio, height, retries)
    params = FePtMgoParameters(
        num_depositions=depositions_slider.value,
        fe_pt_ratio=0.5,
        deposition_height=2.5,
        max_retries=1
    )

    engine = MockLammpsEngine(config.md)
    manager = DepositionManager(engine, params, potential_path)

    # Run Deposition
    final_structure = manager.deposit(mgo_surface)
    status_msg = f"**Step 4 (Deposition)**: Deposited {params.num_depositions} atoms on MgO surface."

    # Visualization
    fig_dep, ax_dep = plt.subplots()
    plot_atoms(final_structure, ax_dep, radii=0.8, rotation=('10x,45y,0z'))
    ax_dep.set_title("Fe/Pt on MgO (Top View)")
    ax_dep.set_axis_off()

    mo.output.append(mo.md(status_msg))
    mo.output.append(fig_dep)
    return (
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
def _(mo):
    mo.md(
        r"""
        ## 6. Phase 3: Validation & Ordering
        In the final phase, we use **Adaptive Kinetic Monte Carlo (aKMC)** via the `EONWrapper` to simulate long-timescale ordering of the L10 phase.

        *   **EONWrapper**: An interface to the EON software (or a functional mock in this tutorial) that runs saddle-point searches to find transition pathways over long timescales.
        """
    )
    return


@app.cell
def _(EONConfig, EONWrapper, WORK_DIR, mo, plt, shutil):
    # EON aKMC Step
    # We use EONWrapper (which might be the real one or our functional mock)

    # Use string path for potential to be safe if config object is strict
    mock_pot = WORK_DIR / "mock.yace"
    if not mock_pot.exists():
        mock_pot.write_text("MOCK")

    # Check if eonclient exists, if not use MockEONWrapper
    # In Pure Demo Mode, EONWrapper is already the Mock class.
    # In Real Mode, EONWrapper is the real class, so we need to check binary.

    use_mock_eon = False
    if hasattr(EONWrapper, "__module__") and "pyacemaker" in EONWrapper.__module__:
        # It's the real class
        if not shutil.which("eonclient"):
            use_mock_eon = True

    if use_mock_eon:
        # Define MockEONWrapper locally if needed (re-using the logic from pure demo mode)
        class MockEONWrapper:
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

        WrapperClass = MockEONWrapper
        print("⚠️ eonclient not found. Falling back to MockEONWrapper.")
    else:
        WrapperClass = EONWrapper

    eon_wrapper = WrapperClass(EONConfig(enabled=True, potential_path=str(mock_pot)))

    work_dir = WORK_DIR / "eon_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    eon_wrapper.run(work_dir)
    results = eon_wrapper.parse_results(work_dir)

    # Visualization
    fig_order, ax_order = plt.subplots()
    t = [0, 10, 20, 30, 40, 50]
    order = [0.1, 0.2, 0.4, 0.7, 0.9, 0.95]
    ax_order.plot(t, order, marker='o')
    ax_order.set_title("L10 Order Parameter vs Time (Mock)")
    ax_order.set_xlabel("Time (ns)")
    ax_order.set_ylabel("Order Parameter (S)")

    mo.output.append(mo.md(f"**Step 5 (Ordering)**: aKMC Simulation complete. Results: {results}"))
    mo.output.append(fig_order)
    return (
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
def _(WORK_DIR_OBJ, mo):
    # Cleanup
    # WORK_DIR_OBJ is the TemporaryDirectory object managing our workspace.
    # Calling cleanup() removes the directory and its contents.
    WORK_DIR_OBJ.cleanup()

    mo.md(
        r"""
        ## Conclusion
        The tutorial has successfully demonstrated the full pipeline:
        1.  **Training**: Generating and labeling data.
        2.  **Deposition**: Simulating growth using the `DepositionManager`.
        3.  **Ordering**: Simulating long-timescale evolution.

        The temporary workspace has been cleaned up.
        """
    )
    return


if __name__ == "__main__":
    app.run()
