import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

@app.cell
def __(mo):
    mo.md(
        """
        # PYACEMAKER Next Generation Architecture Tutorial
        This notebook demonstrates the five key user scenarios of PyAceMaker.
        """
    )

@app.cell
def __():
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import marimo as mo
    import numpy as np
    from ase import Atoms

    from pyacemaker.core.engine import LammpsEngine
    from pyacemaker.core.oracle import DFTManager, MACEManager, TieredOracle
    from pyacemaker.core.trainer import FinetuneManager, PacemakerTrainer
    from pyacemaker.domain_models.md import MDConfig
    from pyacemaker.domain_models.training import TrainingConfig
    from pyacemaker.domain_models.workflow import (
        ActiveLearningThresholds,
        CutoutConfig,
        DistillationConfig,
        LoopStrategyConfig,
        WorkflowConfig,
    )
    from pyacemaker.utils.extraction import extract_intelligent_cluster

    # Enable mock mode
    mock_mode = True
    return (
        ActiveLearningThresholds,
        Atoms,
        CutoutConfig,
        DFTManager,
        DistillationConfig,
        FinetuneManager,
        LammpsEngine,
        LoopStrategyConfig,
        MACEManager,
        MDConfig,
        MagicMock,
        PacemakerTrainer,
        Path,
        TieredOracle,
        TrainingConfig,
        WorkflowConfig,
        extract_intelligent_cluster,
        mo,
        mock_mode,
        np,
        patch,
    )

@app.cell
def __(mo):
    mo.md(
        """
        ## Section 1: Setup and Configuration
        Defining the `WorkflowConfig` and initializing the environment.
        """
    )

@app.cell
def __(DistillationConfig, WorkflowConfig, mo):
    config = WorkflowConfig(
        max_iterations=1, distillation=DistillationConfig(enable=True, uncertainty_threshold=0.05)
    )

    mo.md(f"Initialized configuration: Distillation enabled is **{config.distillation.enable}**")
    return config,

@app.cell
def __(mo):
    mo.md(
        """
        ## Section 2: Zero-Shot Distillation (Scenario 1 & 2)
        Demonstrating Phase 1 and Validator output.
        """
    )

@app.cell
def __(Atoms, MACEManager, mo):
    # 1. MACE evaluates structures
    mace_manager = MACEManager("model")

    atoms1 = Atoms("Fe", cell=[2, 2, 2], pbc=True)
    atoms2 = Atoms("Pt", cell=[2, 2, 2], pbc=True)

    results = list(mace_manager.compute(iter([atoms1, atoms2])))

    # 2. Only structures below threshold are extracted
    for atoms in results:
        c_gamma = atoms.get_array("c_gamma")
        assert (c_gamma <= 0.1).all()  # MACE mock produces up to 0.1

    mo.md(f"Successfully evaluated {len(results)} structures via Zero-Shot Distillation without calling DFT.")
    return atoms1, atoms2, atoms, c_gamma, mace_manager, results

@app.cell
def __(mo):
    mo.md(
        """
        ## Section 3: The Active Learning Event (Scenario 3)
        Injecting a defect, observing the two-tier threshold ignore thermal noise, and triggering an intelligent cutout. Visualizing the passivated cluster.
        """
    )

@app.cell
def __(
    ActiveLearningThresholds,
    Atoms,
    CutoutConfig,
    DFTManager,
    MACEManager,
    MagicMock,
    TieredOracle,
    extract_intelligent_cluster,
    mo,
    np,
    patch,
):
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02)
    cutout_config = CutoutConfig(core_radius=3.0, buffer_radius=2.0)

    mace_manager_tiered = MACEManager("model")
    dft_manager = MagicMock(spec=DFTManager)

    oracle = TieredOracle(mace_manager_tiered, dft_manager, thresholds)

    # 1. Thermal Noise Spike (handled by engine mock logic previously, here we test Oracle fallback)
    # The oracle evaluates a structure. MACE mock yields max_g around 0.1
    defect_atoms = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10])

    with patch("pyacemaker.core.oracle.np.random.uniform", return_value=np.array([0.1, 0.1])):
        gen = oracle.compute(iter([defect_atoms]))
        _result = next(gen)

    # max_g = 0.1 > 0.05, so it falls back to DFT
    dft_manager.compute.assert_called()

    # 2. Extraction of Epicenter
    target_atoms = [0]
    cluster = extract_intelligent_cluster(defect_atoms, target_atoms, cutout_config)

    # Check physical repair
    weights = cluster.get_array("force_weight")
    assert 1.0 in weights

    symbols = cluster.get_chemical_symbols()
    assert len(symbols) > 0

    mo.md(f"Successfully extracted an intelligent cluster with {len(symbols)} atoms and passivated dangling bonds.")
    return (
        cluster,
        cutout_config,
        defect_atoms,
        dft_manager,
        gen,
        mace_manager_tiered,
        oracle,
        symbols,
        target_atoms,
        thresholds,
        weights,
    )

@app.cell
def __(mo):
    mo.md(
        """
        ## Section 4: Incremental Update and Resume (Scenario 4)
        Executing the mock fine-tuning and observing the seamless continuation of the MD step counter.
        """
    )

@app.cell
def __(
    Atoms,
    FinetuneManager,
    LammpsEngine,
    LoopStrategyConfig,
    MDConfig,
    MagicMock,
    PacemakerTrainer,
    Path,
    TrainingConfig,
    mo,
    np,
    patch,
):
    # Setup temporary directory and files
    import tempfile as _tempfile1
    tmp_dir = _tempfile1.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    dataset_path = tmp_path / "dataset.xyz"
    dataset_path.touch()

    # 1. Finetune MACE
    finetune_mgr = FinetuneManager()
    awakened_model = finetune_mgr.finetune(dataset_path)

    # 2. ACE Incremental Update
    t_config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="test_pot.yace",
        delta_learning=True,
        elements=["Fe", "Pt"],
        seed=123,
        max_iterations=500,
        batch_size=20,
    )
    trainer = PacemakerTrainer(t_config)
    strategy = LoopStrategyConfig(replay_buffer_size=100)

    with patch.object(trainer, "train") as mock_train:
        mock_train.return_value = tmp_path / "test_pot.yace"
        new_pot = trainer.incremental_train(dataset_path, strategy, initial_potential="init.yace")

    # 3. Seamless Resume
    md_config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=5000, fix_halt=True
    )

    with patch("pyacemaker.core.engine.LammpsDriver") as mock_driver_class:
        driver_instance = mock_driver_class.return_value
        driver_instance.extract_variable.side_effect = lambda name: {
            "pe": -100.0,
            "step": 2000,
            "max_g": 0.01,
            "temp": 300.0,
            "halted": 0.0,
        }.get(name, 0.0)

        driver_instance.get_forces.return_value = np.zeros((1, 3))
        driver_instance.get_stress.return_value = np.zeros(6)

        script_content = []

        def capture_run(path: str) -> None:
            script_content.append(Path(path).read_text())

        driver_instance.run_file.side_effect = capture_run

        engine = LammpsEngine(md_config)
        md_atoms = Atoms("Fe", cell=[10, 10, 10], pbc=True)
        pot_path = tmp_path / "test_pot.yace"
        pot_path.touch()

        # Resume from step 1500 (halted earlier)
        engine.run(md_atoms, pot_path, resume_from_step=1500)

    resume_success = len(script_content) == 1 and "Resuming from step 1500" in script_content[0]

    # Clean up
    tmp_dir.cleanup()

    mo.md(f"Finetuned model to **{awakened_model}**, generated incremental update, and resumed MD seamlessly: **{resume_success}**.")
    return (
        awakened_model,
        capture_run,
        dataset_path,
        driver_instance,
        engine,
        finetune_mgr,
        md_atoms,
        md_config,
        mock_driver_class,
        mock_train,
        new_pot,
        pot_path,
        resume_success,
        script_content,
        strategy,
        t_config,
        tmp_dir,
        tmp_path,
        trainer,
    )

@app.cell
def __(mo):
    mo.md(
        """
        ## Section 5: State Resilience (Scenario 5)
        Simulating a crash and demonstrating checkpoint recovery.
        """
    )

@app.cell
def __(Path, mo):
    import tempfile as _tempfile2

    from pyacemaker.core.state_manager import StateManager
    from pyacemaker.core.loop import LoopStatus
    from pyacemaker.logger import setup_logger
    from pyacemaker.domain_models.logging import LoggingConfig

    tmp_dir2 = _tempfile2.TemporaryDirectory()
    state_file = Path(tmp_dir2.name) / "state.json"

    logger = setup_logger(LoggingConfig(level="INFO"), "tutorial")
    sm = StateManager(state_file, logger)

    # Save a state with some context
    sm.state.iteration = 5
    sm.state.status = LoopStatus.HALTED
    sm.save()

    # Simulate a crash by reloading from the file
    sm_recovered = StateManager(state_file, logger)
    sm_recovered.load()
    state = sm_recovered.state

    tmp_dir2.cleanup()

    mo.md(f"Recovered state: Iteration {state.iteration}, Status: {state.status}")
    return sm, sm_recovered, state, state_file, tmp_dir2, LoopStatus

if __name__ == "__main__":
    app.run()
