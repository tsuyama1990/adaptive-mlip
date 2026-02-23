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
        elements=["H"],
        seed=123,
        max_iterations=500,
        batch_size=20
    )

@pytest.fixture
def trainer(config: TrainingConfig) -> PacemakerTrainer:
    return PacemakerTrainer(config)

def test_train_success_config_generation(trainer: PacemakerTrainer, tmp_path: Path) -> None:
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

        # Verify generated config structure matches default PacemakerConfig
        args, _ = mock_dump.call_args
        generated_config = args[0]
        assert generated_config["potential"]["elements"] == ["H"]
        assert generated_config["seed"] == 123
        assert generated_config["fit"]["optimizer"] == "BFGS"
        assert generated_config["fit"]["maxiter"] == 500
        assert generated_config["backend"]["batch_size"] == 20
        # Check defaults from PacemakerConfig
        assert generated_config["potential"]["embeddings"]["H"]["npot"] == "FinnisSinclair"
        assert generated_config["fit"]["loss"]["kappa"] == 0.3

def test_train_file_not_found(trainer: PacemakerTrainer) -> None:
    with pytest.raises(TrainerError, match="Training data not found"):
        trainer.train("non_existent.xyz")

def test_train_process_failure(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    data_path = tmp_path / "train.xyz"
    data_path.touch()

    with patch("subprocess.run") as mock_run:
        # Simulate pace_train failure via CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(1, "pace_train", stderr="OOM")

        with pytest.raises(TrainerError, match="Training failed"):
            trainer.train(data_path)
