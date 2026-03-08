import marimo

__generated_with = "0.2.1"
app = marimo.App()


@app.cell
def __():
    import tempfile
    from pathlib import Path

    import marimo as mo
    from ase import Atoms

    from pyacemaker.core.engine import LammpsEngine
    from pyacemaker.core.loop import LoopState, LoopStatus
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

    return (
        mo,
        Path,
        tempfile,
        Atoms,
        WorkflowConfig,
        DistillationConfig,
        ActiveLearningThresholds,
        CutoutConfig,
        LoopStrategyConfig,
        MACEManager,
        DFTManager,
        TieredOracle,
        extract_intelligent_cluster,
        FinetuneManager,
        PacemakerTrainer,
        LoopState,
        LoopStatus,
        LammpsEngine,
        MDConfig,
        TrainingConfig,
    )


@app.cell
def __(
    WorkflowConfig,
    DistillationConfig,
    ActiveLearningThresholds,
    CutoutConfig,
    LoopStrategyConfig,
    mo,
):
    mo.md(
        """
        # PyAceMaker NextGen UAT & Tutorial
        This notebook demonstrates the capabilities of the PyAceMaker Next Generation Architecture.
        """
    )

    config = WorkflowConfig(
        max_iterations=5,
        distillation=DistillationConfig(
            enable=True,
            uncertainty_threshold=0.05,
        ),
        cutout=CutoutConfig(
            core_radius=4.0,
            buffer_radius=3.0,
        ),
        loop_strategy=LoopStrategyConfig(
            thresholds=ActiveLearningThresholds(
                threshold_call_dft=0.05,
                threshold_add_train=0.02,
                smooth_steps=3,
            )
        ),
    )

    return (config,)


@app.cell
def __(Atoms, MACEManager, config, mo):
    mo.md("## Zero-Shot Distillation (Scenarios 1 & 2)")

    # Create dummy structures
    structures = [
        Atoms("Fe", positions=[[0, 0, 0]], cell=[2.8, 2.8, 2.8], pbc=True),
        Atoms("Pt", positions=[[0, 0, 0]], cell=[3.9, 3.9, 3.9], pbc=True),
    ]

    # Initialize MACE Oracle
    mace_manager = MACEManager("mace-mp-0-medium")

    # Evaluate structures
    results = list(mace_manager.compute(iter(structures)))

    # Filter based on threshold
    threshold = config.distillation.uncertainty_threshold
    confident_structures = []

    # Check max_g or max c_gamma
    for res in results:
        # Mock MACE gives c_gamma in arrays
        _c_gamma = res.get_array("c_gamma")
        if _c_gamma.max() <= threshold:
            confident_structures.append(res)

    # Mock fallback, in UAT the MACE mock returns values between 0.01 and 0.1
    # We just ensure the logic works without crashing

    return (
        structures,
        mace_manager,
        results,
        confident_structures,
    )


@app.cell
def __(Atoms, extract_intelligent_cluster, config, mo):
    mo.md("## The Active Learning Event (Scenario 3)")

    # 1. Simulate an MD Halt due to uncertainty > threshold_call_dft
    # Say max_g = 0.1 > config.loop_strategy.thresholds.threshold_call_dft (0.05)

    halt_structure = Atoms(
        "Fe4",
        positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        cell=[10, 10, 10],
    )

    # Simulate uncertainty array
    import numpy as np

    c_gamma = np.array([0.01, 0.1, 0.01, 0.01])
    halt_structure.new_array("c_gamma", c_gamma)

    # 2. Identify epicenters (atoms exceeding threshold_add_train)
    threshold_add_train = config.loop_strategy.thresholds.threshold_add_train
    target_atoms = np.where(c_gamma > threshold_add_train)[0].tolist()

    # 3. Extract intelligent cluster
    cluster = extract_intelligent_cluster(halt_structure, target_atoms, config.cutout)

    # 4. Verify physical repair via force weights
    weights = cluster.get_array("force_weight")

    # Ensure there is at least one core atom (weight 1.0)
    has_core = 1.0 in weights

    return (
        halt_structure,
        target_atoms,
        cluster,
        weights,
        has_core,
    )


@app.cell
def __(
    FinetuneManager,
    PacemakerTrainer,
    TrainingConfig,
    LoopStrategyConfig,
    Path,
    tempfile,
    mo,
):
    mo.md("## Incremental Update & Resume (Scenario 4)")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Finetune MACE
        dataset_path = temp_path / "dataset.xyz"
        dataset_path.touch()

        finetune_mgr = FinetuneManager()
        awakened_model = finetune_mgr.finetune(dataset_path)

        # 2. ACE Incremental Update
        training_config = TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=2,
            output_filename="test_pot.yace",
            delta_learning=True,
            elements=["Fe"],
            seed=123,
            max_iterations=5,
            batch_size=20,
        )

        # 3. Seamless Resume logic execution simulated

        # A mocked train to simulate incremental update success
        class MockPacemakerTrainer(PacemakerTrainer):
            def train(self, data_path, init_pot=None):
                output_path = temp_path / self.config.output_filename
                output_path.touch()
                return output_path

        trainer = MockPacemakerTrainer(training_config)

        _strategy_config = LoopStrategyConfig()
        new_pot = trainer.incremental_train(dataset_path, strategy_config=_strategy_config)

    return (
        finetune_mgr,
        awakened_model,
        training_config,
        trainer,
        new_pot,
    )


@app.cell
def __(LoopState, LoopStatus, Path, tempfile, mo):
    mo.md("## State Resilience (Scenario 5)")

    with tempfile.TemporaryDirectory() as _temp_dir:
        _temp_path = Path(_temp_dir)
        state_file = _temp_path / "pyacemaker_state.json"

        # 1. Create and Save State
        original_state = LoopState(
            iteration=42,
            status=LoopStatus.HALTED,
            current_potential=None,
        )

        # Save uses atomic write and file locking to prevent corruption
        original_state.save(state_file)

        # 2. Simulate Crash and Recovery
        # Load state after restart
        recovered_state = LoopState.load(state_file)

        # Verify it matches
        assert recovered_state.iteration == 42
        assert recovered_state.status == LoopStatus.HALTED

    return (
        original_state,
        recovered_state,
    )


if __name__ == "__main__":
    app.run()
