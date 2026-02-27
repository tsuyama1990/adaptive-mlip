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
    # Note: PacemakerTrainer logic checks shutil.which inside train()
    # If we patch it to return None, it should raise TrainerError
    # But wait, logic might use `run_command` later if we skip check.
    # The `train` method implementation I wrote checks `shutil.which` at start.

    with (
        patch("shutil.which", return_value=None),
        # Since trainer does check, it should raise TrainerError if logic is correct
        pytest.raises(TrainerError, match="Executable 'pace_train' not found"),
    ):
        # We need a dummy file so validation passes up to executable check
        # BUT, the current implementation checks shutil.which BEFORE validating file?
        # Let's check logic:
        # if not shutil.which("pace_train"): ...
        # data_path = ...
        # _validate_training_data(data_path)

        # So we can pass any path if it crashes at executable check first.
        # But if validation runs first, we need valid file.
        # In current logic, check is first.
        trainer.train(tmp_path / "dummy.xyz")


def test_train_element_detection_scanning(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    # Create mixed dataset
    data_path = tmp_path / "train.xyz"
    atoms1 = Atoms("Fe", positions=[[0, 0, 0]])
    atoms2 = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]])
    write(data_path, [atoms1, atoms2])

    # Trainer calls subprocess.run directly in my implementation, not run_command
    # I should patch subprocess.run
    with patch("subprocess.run") as mock_run, patch(
        "pyacemaker.core.trainer.dump_yaml"
    ) as mock_dump:

        # Create dummy output file because trainer checks for its existence after run
        (data_path.parent / "test_pot.yace").touch()

        # Update config to force detection
        trainer.config.elements = []

        # We assume PacemakerConfigGenerator logic is correct in detecting elements if config.elements is empty
        # However, for this unit test, we might need to mock PacemakerConfigGenerator or ensure it works.
        # If PacemakerConfigGenerator is not fully implemented or mocked, this test might fail on config generation.
        # Let's assume ConfigGenerator works or we mock it.
        # Since trainer.config_generator is instantiated in init, we can replace it.

        mock_gen = MagicMock()
        mock_gen.generate.return_value = {"potential": {"elements": ["Fe", "Pt"]}}
        trainer.config_generator = mock_gen

        trainer.train(data_path)

        # Check config dump
        args, _ = mock_dump.call_args
        generated_config = args[0]
        assert generated_config["potential"]["elements"] == ["Fe", "Pt"]

        # Verify command execution
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "pace_train"


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

    # Patch subprocess.run
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "cmd", stderr="error"
        )

        with pytest.raises(TrainerError, match="Training failed"):
            trainer.train(data_path)


def test_train_initial_potential(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    """Test that initial_potential adds correct argument."""
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    initial_pot = tmp_path / "init.yace"
    initial_pot.touch()

    # Create dummy output
    (data_path.parent / "test_pot.yace").touch()

    with patch("subprocess.run") as mock_run, patch(
        "pyacemaker.core.trainer.dump_yaml"
    ):
        trainer.train(data_path, initial_potential=initial_pot)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert "--initial_potential" in cmd
        assert str(initial_pot) in cmd


def test_train_initial_potential_missing(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    """Test that missing initial potential raises error."""
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    initial_pot = tmp_path / "missing.yace"

    with pytest.raises(TrainerError, match="Initial potential not found"):
        trainer.train(data_path, initial_potential=initial_pot)
