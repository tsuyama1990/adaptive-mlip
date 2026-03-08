from typing import Any

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def intro() -> tuple[Any, ...]:
    import marimo as mo

    mo.md(
        """
        # PYACEMAKER Next Generation Architecture Tutorial
        This notebook demonstrates the five key user scenarios of PyAceMaker.
        """
    )
    return (mo,)


@app.cell
def setup_environment() -> tuple[Any, ...]:
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

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
        tempfile,
        np,
        patch,
    )


@app.cell
def run_setup(
    DistillationConfig: type, WorkflowConfig: type, mo: object
) -> tuple[Any, ...]:
    config = WorkflowConfig(
        max_iterations=1, distillation=DistillationConfig(enable=True, uncertainty_threshold=0.05)
    )

    if hasattr(mo, "md"):
        mo.md(f"Initialized configuration: Distillation enabled is **{config.distillation.enable}**")
    return (config,)


@app.cell
def run_distillation(
    Atoms: type, DistillationConfig: type, MACEManager: type, WorkflowConfig: type, mo: object
) -> tuple[Any, ...]:
    _config2 = WorkflowConfig(
        max_iterations=1, distillation=DistillationConfig(enable=True, uncertainty_threshold=0.05)
    )
    # 1. MACE evaluates structures
    mace_manager = MACEManager(_config2.distillation.mace_model_path)

    atoms1 = Atoms("Fe", cell=[2, 2, 2], pbc=True)
    atoms2 = Atoms("Pt", cell=[2, 2, 2], pbc=True)

    results = list(mace_manager.compute(iter([atoms1, atoms2])))

    # 2. Only structures below threshold are extracted
    for d_atoms in results:
        c_gamma = d_atoms.get_array("c_gamma")
        if not (c_gamma <= 0.1).all():
            _msg1 = "MACE mock produced unexpected output."
            raise ValueError(_msg1)

    if hasattr(mo, "md"):
        mo.md(
            f"Successfully evaluated {len(results)} structures via Zero-Shot Distillation without calling DFT."
        )
    return mace_manager, results


@app.cell
def run_active_learning_event(
    ActiveLearningThresholds: type,
    Atoms: type,
    CutoutConfig: type,
    DFTManager: type,
    DistillationConfig: type,
    MACEManager: type,
    MagicMock: type,
    TieredOracle: type,
    WorkflowConfig: type,
    extract_intelligent_cluster: type,
    mo: object,
    np: Any,
    patch: Any,
) -> tuple[Any, ...]:
    _config3 = WorkflowConfig(
        max_iterations=1, distillation=DistillationConfig(enable=True, uncertainty_threshold=0.05)
    )
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02)
    cutout_config = CutoutConfig(core_radius=3.0, buffer_radius=2.0)

    mace_manager_tiered = MACEManager(_config3.distillation.mace_model_path)
    dft_manager = MagicMock(spec=DFTManager)

    oracle = TieredOracle(mace_manager_tiered, dft_manager, thresholds)

    defect_atoms = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10])

    with patch("pyacemaker.core.oracle.np.random.uniform", return_value=np.array([0.1, 0.1])):
        gen = oracle.compute(iter([defect_atoms]))
        _result = next(gen)

    dft_manager.compute.assert_called()

    target_atoms = [0]
    cluster = extract_intelligent_cluster(defect_atoms, target_atoms, cutout_config)

    weights = cluster.get_array("force_weight")
    if 1.0 not in weights:
        _msg2 = "Core atoms missing in intelligent cluster."
        raise ValueError(_msg2)

    symbols = cluster.get_chemical_symbols()
    if len(symbols) <= 0:
        _msg3 = "Passivation failed."
        raise ValueError(_msg3)

    if hasattr(mo, "md"):
        mo.md(
            f"Successfully extracted an intelligent cluster with {len(symbols)} atoms and passivated dangling bonds."
        )
    return oracle, cluster


@app.cell
def run_incremental_update_and_resume(
    Atoms: type,
    FinetuneManager: type,
    LammpsEngine: type,
    LoopStrategyConfig: type,
    MDConfig: type,
    PacemakerTrainer: type,
    Path: Any,
    TrainingConfig: type,
    mo: object,
    np: Any,
    patch: Any,
    tempfile: Any,
) -> tuple[Any, ...]:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_path = Path(tmp_dir_name)
        dataset_path = tmp_path / "dataset.xyz"
        dataset_path.touch()

        finetune_mgr = FinetuneManager()
        awakened_model = finetune_mgr.finetune(dataset_path)

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
            mock_train.return_value = tmp_path / t_config.output_filename
            _new_pot = trainer.incremental_train(
                dataset_path, strategy, initial_potential="init.yace"
            )

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

            def local_capture_run(path: str) -> None:
                script_content.append(Path(path).read_text())

            driver_instance.run_file.side_effect = local_capture_run

            engine = LammpsEngine(md_config)
            md_atoms = Atoms("Fe", cell=[10, 10, 10], pbc=True)
            pot_path = tmp_path / t_config.output_filename
            pot_path.touch()

            engine.run(md_atoms, pot_path, resume_from_step=1500)

        resume_success = len(script_content) == 1 and "Resuming from step 1500" in script_content[0]

    if hasattr(mo, "md"):
        mo.md(
            f"Finetuned model to **{awakened_model}**, generated incremental update, and resumed MD seamlessly: **{resume_success}**."
        )
    return awakened_model, resume_success


@app.cell
def run_state_resilience(Path: Any, mo: object, tempfile: Any) -> tuple[Any, ...]:
    from pyacemaker.core.loop import LoopStatus
    from pyacemaker.core.state_manager import StateManager
    from pyacemaker.domain_models.logging import LoggingConfig
    from pyacemaker.logger import setup_logger

    with tempfile.TemporaryDirectory() as tmp_dir2_name:
        state_file = Path(tmp_dir2_name) / "state.json"

        logger = setup_logger(LoggingConfig(level="INFO"), "tutorial")
        sm = StateManager(state_file, logger)

        sm.state.iteration = 5
        sm.state.status = LoopStatus.HALTED
        sm.save()

        sm_recovered = StateManager(state_file, logger)
        sm_recovered.load()
        state = sm_recovered.state

    if hasattr(mo, "md"):
        mo.md(f"Recovered state: Iteration {state.iteration}, Status: {state.status}")
    return state.iteration, state.status


if __name__ == "__main__":
    app.run()
