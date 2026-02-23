from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.exceptions import TrainerError
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


@pytest.fixture
def config() -> TrainingConfig:
    return TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="test_pot.yace",
        delta_learning=True
    )

@pytest.fixture
def trainer(config: TrainingConfig) -> PacemakerTrainer:
    return PacemakerTrainer(config)


def test_train_success(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    # Create dummy training data
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]))

    with patch("subprocess.run") as mock_run, \
         patch("pyacemaker.core.trainer.dump_yaml") as mock_dump:

        mock_run.return_value = MagicMock(returncode=0)

        # Mock file existence check for output (which is usually created by subprocess)
        # We can mock Path.exists or just assume the Trainer checks it after
        with patch.object(Path, "exists", return_value=True):
            result = trainer.train(data_path)

        assert result.name == "test_pot.yace"
        mock_run.assert_called()
        # Check if config.yaml was generated
        mock_dump.assert_called()

def test_train_file_not_found(trainer: PacemakerTrainer) -> None:
    with pytest.raises(TrainerError, match="Training data not found"):
        trainer.train("non_existent.xyz")

def test_train_process_failure(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    data_path = tmp_path / "train.xyz"
    data_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = RuntimeError("Process crashed")

        with pytest.raises(TrainerError):
            trainer.train(data_path)
