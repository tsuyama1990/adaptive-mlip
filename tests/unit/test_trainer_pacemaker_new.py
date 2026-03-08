from pathlib import Path
from typing import Any

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.exceptions import TrainerError
from pyacemaker.core.trainer import FinetuneManager, PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.domain_models.workflow import LoopStrategyConfig


def test_finetune_manager(tmp_path: Path) -> None:
    manager = FinetuneManager()

    dft_data = tmp_path / "dft.xyz"
    write(dft_data, Atoms("H2", positions=[[0,0,0], [0,0,1]]))

    model_path = manager.finetune(dft_data)
    assert model_path.exists()
    assert model_path.name == "finetuned_mace.model"

def test_finetune_manager_missing_data(tmp_path: Path) -> None:
    manager = FinetuneManager()
    with pytest.raises(TrainerError, match="DFT data not found"):
        manager.finetune(tmp_path / "missing.xyz")


@pytest.fixture
def trainer(tmp_path: Path) -> PacemakerTrainer:
    config = TrainingConfig(
        potential_type="ace", cutoff_radius=5.0, max_basis_size=10,
        output_filename="pot.yace", delta_learning=True, elements=["Fe", "H"]
    )
    return PacemakerTrainer(config)

def test_get_replay_buffer(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    hist_data = tmp_path / "hist.xyz"

    # Write 5 structures
    structures = [Atoms(f"H{i}") for i in range(1, 6)]
    write(hist_data, structures)

    # Get 3
    buffer = trainer.get_replay_buffer(hist_data, 3)
    assert len(buffer) == 3

    # Get more than exist
    buffer = trainer.get_replay_buffer(hist_data, 10)
    assert len(buffer) == 5

    # Empty
    buffer = trainer.get_replay_buffer(tmp_path / "missing.xyz", 3)
    assert len(buffer) == 0

def test_incremental_train(trainer: PacemakerTrainer, tmp_path: Path, mocker: "Any") -> None:
    new_data = tmp_path / "new.xyz"
    hist_data = tmp_path / "hist.xyz"

    write(new_data, [Atoms("H")])
    write(hist_data, [Atoms("H2"), Atoms("H3")])

    strategy = LoopStrategyConfig(replay_buffer_size=1)

    mocker.patch.object(trainer, "train", return_value=Path("mock.yace"))

    # Check that train is called with blended data
    res = trainer.incremental_train(new_data, hist_data, None, strategy)

    assert res.name == "mock.yace"
    trainer.train.assert_called_once()
    blended_path = trainer.train.call_args[0][0]

    # blended data should have 1 + 1 = 2 structures
    from ase.io import read
    blended = read(blended_path, index=":")
    assert len(blended) == 2
