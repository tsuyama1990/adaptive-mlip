from pathlib import Path

import pytest

from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


def test_pacemaker_trainer_train_success(mock_training_config: TrainingConfig, tmp_path: Path) -> None:
    trainer = PacemakerTrainer(mock_training_config)

    # Create dummy training data
    data_file = tmp_path / "training.pckl"
    data_file.touch()

    result = trainer.train(data_file)

    assert isinstance(result, Path)
    assert result.name == "potential.yace"
    assert result.parent == tmp_path

def test_pacemaker_trainer_train_missing_file(mock_training_config: TrainingConfig, tmp_path: Path) -> None:
    trainer = PacemakerTrainer(mock_training_config)

    with pytest.raises(FileNotFoundError):
        trainer.train(tmp_path / "missing.pckl")
