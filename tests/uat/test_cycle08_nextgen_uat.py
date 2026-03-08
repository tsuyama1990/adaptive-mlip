from pathlib import Path

from ase import Atoms

from pyacemaker.domain_models.workflow import CutoutConfig


def test_scenario_08_01_phase1_distillation(tmp_path: Path) -> None:
    """
    Scenario UAT-PHASE1-001: Phase 1 - Zero-Shot Distillation
    """
    # Use Pydantic objects directly to verify logic (mock testing config setup)
    from pyacemaker.domain_models.workflow import DistillationConfig

    config = DistillationConfig(
        enable=True,
        mace_model_path="mock-mace-model",
        uncertainty_threshold=0.05,
        sampling_count=100
    )

    assert config.enable is True
    assert config.sampling_count == 100

def test_scenario_08_02_phase3_cutout(tmp_path: Path) -> None:
    """
    Scenario UAT-PHASE3-001: Intelligent Cutout and Passivation
    """
    from pyacemaker.utils.extraction import extract_intelligent_cluster

    # Large structure
    atoms = Atoms("Fe10", positions=[[i, 0, 0] for i in range(10)], cell=[20,20,20], pbc=True)

    config = CutoutConfig(core_radius=1.5, buffer_radius=1.5, pre_relax=False, passivation=True)

    # Target center atom (e.g. index 5)
    cluster = extract_intelligent_cluster(atoms, [5], config)

    assert len(cluster) > 0
    # Must have force_weights
    assert "force_weight" in cluster.arrays

from typing import Any


def test_scenario_08_03_phase4_hierarchical_finetuning(tmp_path: Path, mocker: "Any") -> None:
    """
    Scenario UAT-PHASE4-001: Hierarchical Finetuning & Seamless Resume
    """
    from ase.io import write

    from pyacemaker.core.trainer import PacemakerTrainer
    from pyacemaker.domain_models.training import TrainingConfig
    from pyacemaker.domain_models.workflow import LoopStrategyConfig

    # Dummy data
    dft_data = tmp_path / "dft.xyz"
    hist_data = tmp_path / "hist.xyz"
    write(dft_data, Atoms("Fe"))
    write(hist_data, [Atoms("Fe")])

    config = TrainingConfig(
        potential_type="ace", cutoff_radius=5.0, max_basis_size=10,
        output_filename="pot.yace", delta_learning=True, elements=["Fe"]
    )
    trainer = PacemakerTrainer(config)

    # Mock trainer.train to prevent subprocess call
    mocker.patch.object(trainer, "train", return_value=Path("mock.yace"))

    strategy = LoopStrategyConfig(replay_buffer_size=1)

    res = trainer.incremental_train(dft_data, hist_data, None, strategy)

    # Should complete without error
    assert str(res) == "mock.yace"
    trainer.train.assert_called_once()
