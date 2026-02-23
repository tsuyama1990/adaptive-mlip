import subprocess
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
        delta_learning=True,
        # elements not provided, must be detected
        seed=123,
        max_iterations=500,
        batch_size=20
    )

@pytest.fixture
def trainer(config: TrainingConfig) -> PacemakerTrainer:
    return PacemakerTrainer(config)

def test_train_element_detection_scanning(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    # Create mixed dataset
    data_path = tmp_path / "train.xyz"
    # First frame only Fe
    # Second frame Fe + Pt
    atoms1 = Atoms("Fe", positions=[[0,0,0]])
    atoms2 = Atoms("FePt", positions=[[0,0,0], [1,1,1]])
    write(data_path, [atoms1, atoms2])

    with patch("pyacemaker.core.trainer.run_command") as mock_run, \
         patch("pyacemaker.core.trainer.dump_yaml") as mock_dump:

        mock_run.return_value = MagicMock(returncode=0)

        # Create dummy output
        (tmp_path / "test_pot.yace").touch()

        trainer.train(data_path)

        args, _ = mock_dump.call_args
        generated_config = args[0]
        # Should detect both Fe and Pt even if first frame only has Fe
        assert generated_config["potential"]["elements"] == ["Fe", "Pt"]

def test_train_validation_empty_file(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    data_path = tmp_path / "empty.xyz"
    data_path.touch()

    with pytest.raises(TrainerError, match="is empty"):
        trainer.train(data_path)

def test_train_process_fail_util(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    with patch("pyacemaker.core.trainer.run_command") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with pytest.raises(TrainerError, match="Training failed"):
            trainer.train(data_path)
