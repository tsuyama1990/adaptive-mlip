import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from ase import Atoms
from ase.io import write

from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


@pytest.fixture
def base_training_data(tmp_path: Path) -> Path:
    data_path = tmp_path / "training_data.xyz"
    atoms = [Atoms("H2O"), Atoms("H2O")]
    write(data_path, atoms)
    return data_path

@pytest.fixture
def trainer_config() -> TrainingConfig:
    return TrainingConfig(
        potential_type="pace",
        cutoff_radius=4.0,
        max_basis_size=300,
        output_filename="output.yace",
    )

def test_pacemaker_element_detection(tmp_path: Path, base_training_data: Path, trainer_config: TrainingConfig) -> None:
    trainer = PacemakerTrainer(trainer_config)

    with (
        patch("shutil.which", return_value="/usr/bin/pace_train"),
        patch("pyacemaker.core.trainer.run_command") as mock_run,
    ):
        def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
            input_yaml = Path(cmd[1])
            with input_yaml.open() as f:
                data = yaml.safe_load(f)
                assert data["potential"]["elements"] == ["H", "O"]
            (tmp_path / "output.yace").touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        trainer.train(base_training_data)

def test_pacemaker_config_generation(tmp_path: Path, base_training_data: Path, trainer_config: TrainingConfig) -> None:
    trainer = PacemakerTrainer(trainer_config)

    with (
        patch("shutil.which", return_value="/usr/bin/pace_train"),
        patch("pyacemaker.core.trainer.run_command") as mock_run,
    ):
        def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
            input_yaml = Path(cmd[1])
            with input_yaml.open() as f:
                data = yaml.safe_load(f)
                assert data["cutoff"] == 4.0
            (tmp_path / "output.yace").touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        trainer.train(base_training_data)

def test_pacemaker_command_execution(tmp_path: Path, base_training_data: Path, trainer_config: TrainingConfig) -> None:
    trainer = PacemakerTrainer(trainer_config)
    output_pot_path = tmp_path / "output.yace"

    with (
        patch("shutil.which", return_value="/usr/bin/pace_train"),
        patch("pyacemaker.core.trainer.run_command") as mock_run,
    ):
        def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
            output_pot_path.touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        result = trainer.train(base_training_data)

        assert result == output_pot_path
        assert result.exists()
        mock_run.assert_called_once()


def test_pacemaker_integration_failure_handling(tmp_path: Path) -> None:
    """Test trainer failure handling."""
    data_path = tmp_path / "data.xyz"
    write(data_path, Atoms("He"))

    config = TrainingConfig(potential_type="pace", cutoff_radius=3.0, max_basis_size=100)
    trainer = PacemakerTrainer(config)

    from pyacemaker.core.exceptions import TrainerError

    with (
        patch("shutil.which", return_value="/bin/true"),
        patch("pyacemaker.core.trainer.run_command") as mock_run,
    ):
        # Simulate process failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

        with pytest.raises(TrainerError, match="Training failed with exit code 1"):
            trainer.train(data_path)
