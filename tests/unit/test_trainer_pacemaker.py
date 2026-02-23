import subprocess
from collections.abc import Generator
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


@pytest.fixture
def mock_shutil_which() -> Generator[MagicMock, None, None]:
    with patch("shutil.which") as mock:
        mock.return_value = "/usr/bin/pace_train"
        yield mock


def test_train_missing_executable(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    """Test that missing pace_train raises TrainerError."""
    # Ensure validation fails BEFORE data check or AFTER?
    # Trainer implementation checks executable in executor.run_training, which happens AFTER validation.
    # So we need a valid dummy file for validation to pass.
    # It must also not be empty to pass _validate_training_data
    dummy_file = tmp_path / "dummy.pckl"
    dummy_file.write_text("dummy content")

    with (
        patch("shutil.which", return_value=None),
        pytest.raises(TrainerError, match="Executable 'pace_train' not found"),
    ):
        trainer.train(dummy_file)


def test_train_element_detection_scanning(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    # Create mixed dataset
    data_path = tmp_path / "train.xyz"
    atoms1 = Atoms("Fe", positions=[[0, 0, 0]])
    atoms2 = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]])
    write(data_path, [atoms1, atoms2])

    with patch("pyacemaker.core.executor.run_command") as mock_run, patch(
        "pyacemaker.core.trainer.dump_yaml"
    ) as mock_dump, patch("shutil.which", return_value="/usr/bin/pace_train"):
        mock_run.return_value = MagicMock(returncode=0)

        # Create dummy output
        (tmp_path / "test_pot.yace").touch()

        # Update config to force detection (clear elements)
        trainer.config.elements = None

        trainer.train(data_path)

        args, _ = mock_dump.call_args
        generated_config = args[0]
        # Should detect both Fe and Pt even if first frame only has Fe
        assert generated_config["potential"]["elements"] == ["Fe", "Pt"]
        # Verify structure
        assert "fit" in generated_config
        assert "backend" in generated_config
        assert generated_config["backend"]["evaluator"] == "tensorpot"

        # Verify command execution
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "pace_train"
        assert str(cmd_args[1]).endswith("input.yaml")


def test_train_element_detection_scanning_missing_element(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    """Test that missing elements raise TrainerError."""
    # Create empty-ish dataset (valid file but no atoms with symbols?? ASE usually errors or empty list)
    # Actually iread yields Atoms.
    data_path = tmp_path / "train.xyz"
    # Create a file that ase can read but has no symbols? or empty?
    # Empty file is caught by size check.
    # File with 0 atoms per frame?
    atoms = Atoms()
    write(data_path, atoms)

    with patch("pyacemaker.core.executor.run_command") as mock_run, \
         patch("shutil.which", return_value="/usr/bin/pace_train"):
        mock_run.return_value = MagicMock(returncode=0)
        (tmp_path / "test_pot.yace").touch()
        trainer.config.elements = None

        with pytest.raises(TrainerError, match="No elements detected"):
             trainer.train(data_path)


def test_train_validation_empty_file(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    data_path = tmp_path / "empty.xyz"
    data_path.touch()

    with pytest.raises(TrainerError, match="is empty"):
        trainer.train(data_path)


def test_train_process_fail_util(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    with patch("pyacemaker.core.executor.run_command") as mock_run, \
         patch("shutil.which", return_value="/usr/bin/pace_train"):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "cmd", stderr="error"
        )

        with pytest.raises(TrainerError, match="Training failed"):
            trainer.train(data_path)
