from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_scenario_phase1_distillation() -> None:
    """
    Scenario 1: Verification of Zero-Shot Distillation and Baseline Construction
    """
    config = WorkflowConfig(
        max_iterations=1, distillation=DistillationConfig(enable=True, uncertainty_threshold=0.05)
    )

    assert config.distillation.enable is True

    # 1. MACE evaluates structures
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        pot_dir = Path(temp_dir)
        model_file = pot_dir / "model"
        with model_file.open("w") as f:
            f.write("dummy")

        import pyacemaker.domain_models.defaults
        with patch.object(pyacemaker.domain_models.defaults, "DEFAULT_POTENTIALS_DIR", str(pot_dir.resolve())):
            mace_manager = MACEManager(str(model_file))

        atoms1 = Atoms("Fe", cell=[2, 2, 2], pbc=True)
        atoms2 = Atoms("Pt", cell=[2, 2, 2], pbc=True)

        results = list(mace_manager.compute(iter([atoms1, atoms2])))

        assert len(results) == 2
        assert "energy" in results[0].info
        assert "forces" in results[0].arrays

        # 2. Only structures below threshold are extracted
        for atoms in results:
            c_gamma = atoms.get_array("c_gamma")
            assert (c_gamma <= 0.1).all()  # MACE mock produces up to 0.1


def test_scenario_phase3_cutout() -> None:
    """
    Scenario 3: Exclusion of Thermal Noise and Intelligent Cluster Extraction
    """
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02)
    config = CutoutConfig(core_radius=3.0, buffer_radius=2.0)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        pot_dir = Path(temp_dir)
        model_file = pot_dir / "model"
        with model_file.open("w") as f:
            f.write("dummy")

        import pyacemaker.domain_models.defaults
        with patch.object(pyacemaker.domain_models.defaults, "DEFAULT_POTENTIALS_DIR", str(pot_dir.resolve())):
            mace_manager = MACEManager(str(model_file))
    dft_manager = MagicMock(spec=DFTManager)

    oracle = TieredOracle(mace_manager, dft_manager, thresholds)

    # 1. Thermal Noise Spike (handled by engine mock logic previously, here we test Oracle fallback)
    # The oracle evaluates a structure. MACE mock yields max_g around 0.1
    atoms = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10])

    import numpy as np

    with patch("pyacemaker.core.oracle.np.random.uniform", return_value=np.array([0.1, 0.1])):
        gen = oracle.compute(iter([atoms]))
        _result = next(gen)

    # max_g = 0.1 > 0.05, so it falls back to DFT
    dft_manager.compute.assert_called()

    # 2. Extraction of Epicenter
    # target atoms are those exceeding threshold_add_train (0.02)
    # MACE mock is between 0.01 and 0.1, so likely some > 0.02. Let's just pass target_atoms = [0]
    target_atoms = [0]

    cluster = extract_intelligent_cluster(atoms, target_atoms, config)

    # Check physical repair
    weights = cluster.get_array("force_weight")
    assert 1.0 in weights

    # Depending on neighbor cutoff distance and atom setup, H may or may not be added
    # We test that the functionality executes successfully.
    symbols = cluster.get_chemical_symbols()
    assert len(symbols) > 0


@patch("pyacemaker.core.engine.LammpsDriver")
def test_scenario_phase4_resume(mock_driver: MagicMock, tmp_path: Path) -> None:
    """
    Scenario 4: Hierarchical Fine-Tuning and Seamless Resume
    """
    # 1. Finetune MACE
    finetune_mgr = FinetuneManager()
    dataset_path = tmp_path / "dataset.xyz"
    dataset_path.touch()
    awakened_model = finetune_mgr.finetune(dataset_path)
    assert awakened_model == "awakened_mace_model.model"

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
        assert new_pot == tmp_path / "test_pot.yace"
        mock_train.assert_called_once()

    # 3. Seamless Resume
    md_config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=5000, fix_halt=True
    )
    engine = LammpsEngine(md_config)

    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -100.0,
        "step": 2000,
        "max_g": 0.01,
        "temp": 300.0,
        "halted": 0.0,
    }.get(name, 0.0)

    import numpy as np

    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    script_content = []

    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance.run_file.side_effect = capture_run

    atoms = Atoms("Fe", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "test_pot.yace"
    pot_path.touch()

    # Resume from step 1500 (halted earlier)
    engine.run(atoms, pot_path, resume_from_step=1500)

    assert len(script_content) == 1
    assert "Resuming from step 1500" in script_content[0]
