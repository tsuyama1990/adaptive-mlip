import subprocess
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.exceptions import TrainerError
from pyacemaker.core.trainer import FinetuneManager, PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


@pytest.fixture
def config() -> TrainingConfig:
    return TrainingConfig(
        potential_type="pace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="test_pot.yace",
        delta_learning=True,
        elements=["H"],
        seed=123,
        max_iterations=500,
        batch_size=20,
    )


@pytest.fixture
def trainer(config: TrainingConfig) -> PacemakerTrainer:
    return PacemakerTrainer(config)


@pytest.fixture
def mock_shutil_which() -> Generator[MagicMock, None, None]:
    with patch("shutil.which") as mock:
        mock.return_value = "/usr/bin/pace_train"
        yield mock


def test_train_missing_executable_error(trainer: PacemakerTrainer, tmp_path: Path) -> None:
    """Test that missing pace_train raises TrainerError."""
    trainer.config = TrainingConfig.model_validate(trainer.config)
    dummy_data = tmp_path / "dummy.xyz"
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(TrainerError, match="Executable 'pace_train' not found"),
    ):
        trainer.train(dummy_data)


def test_train_element_detection_scanning(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    trainer.config = TrainingConfig.model_validate(trainer.config)
    # Create mixed dataset
    data_path = tmp_path / "train.xyz"
    atoms1 = Atoms("Fe", positions=[[0, 0, 0]])
    atoms2 = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]])
    write(data_path, [atoms1, atoms2])

    with (
        patch("pyacemaker.core.trainer.run_command") as mock_run,
        patch("pyacemaker.core.trainer.dump_yaml") as mock_dump,
    ):
        # Create dummy output so file check passes
        (data_path.parent / "test_pot.yace").touch()

        # Update config to force detection (clear elements)
        trainer.config.elements = []  # Assuming empty list triggers detection or None?
        # Check config_generator logic: if self.config.elements: return sorted...
        # So we need to set it to empty list or None.
        # Pydantic model might enforce list. Let's check TrainingConfig.
        # But here config is a fixture object.
        # Let's set it to None if type allows, or empty list.
        # Code says: if self.config.elements:
        trainer.config.elements = []

        trainer.train(data_path)

        args, _ = mock_dump.call_args
        generated_config = args[0]
        # Should detect both Fe and Pt even if first frame only has Fe
        assert generated_config["potential"]["elements"] == ["Fe", "Pt"]

        # Verify command execution
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "pace_train"
        assert str(cmd_args[1]).endswith("input.yaml")


def test_train_validation_empty_file(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    trainer.config = TrainingConfig.model_validate(trainer.config)
    data_path = tmp_path / "empty.xyz"
    data_path.touch()

    with pytest.raises(TrainerError, match="is empty"):
        trainer.train(data_path)


def test_train_process_fail_util(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    trainer.config = TrainingConfig.model_validate(trainer.config)
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    with patch("pyacemaker.core.trainer.run_command") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with pytest.raises(TrainerError, match="Training failed"):
            trainer.train(data_path)


def test_train_initial_potential(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    """Test that initial_potential adds correct argument."""
    trainer.config = TrainingConfig.model_validate(trainer.config)
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    initial_pot = tmp_path / "init.yace"
    initial_pot.touch()

    with (
        patch("pyacemaker.core.trainer.run_command") as mock_run,
        patch("pyacemaker.core.trainer.dump_yaml"),
    ):
        # Create dummy output
        (data_path.parent / "test_pot.yace").touch()

        trainer.train(data_path, initial_potential=initial_pot)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert "--initial_potential" in cmd
        assert str(initial_pot) in cmd


def test_finetune_manager_mock(caplog: pytest.LogCaptureFixture) -> None:
    import logging
    caplog.set_level(logging.INFO)
    manager = FinetuneManager(use_mock=True)
    manager.finetune(Path("dummy.xyz"))
    assert "Mock MACE fine-tuning completed" in caplog.text

def test_finetune_manager_no_mock_raises() -> None:
    manager = FinetuneManager(use_mock=False)
    with pytest.raises(RuntimeError, match="MACE model is not installed"):
        manager.finetune(Path("dummy.xyz"))

def test_train_with_replay_buffer(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    trainer.config = TrainingConfig.model_validate(trainer.config)
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H", positions=[[0,0,0]]))

    replay_path = tmp_path / "replay.xyz"
    # Create 10 structures in replay buffer
    replay_atoms = [Atoms("H", positions=[[i,i,i]]) for i in range(10)]
    write(replay_path, replay_atoms)

    with (
        patch("pyacemaker.core.trainer.run_command"),
        patch("pyacemaker.core.trainer.dump_yaml") as mock_dump,
    ):
        (data_path.parent / "test_pot.yace").touch()

        # Limit to 5 replay structures
        trainer.train(data_path, replay_buffer_path=replay_path, replay_buffer_size=5)

        args, _ = mock_dump.call_args
        args[0]

        # Data passed to pacemaker config generator should be the mixed file
        mixed_file = data_path.with_name(f"{data_path.stem}_mixed{data_path.suffix}")
        assert mixed_file.exists()

        from ase.io import read
        mixed_atoms = list(read(mixed_file, index=":"))
        # 1 new + 5 sampled from replay = 6 total
        assert len(mixed_atoms) == 6


def test_train_initial_potential_missing(
    trainer: PacemakerTrainer, tmp_path: Path, mock_shutil_which: MagicMock
) -> None:
    """Test that missing initial potential raises error."""
    trainer.config = TrainingConfig.model_validate(trainer.config)
    data_path = tmp_path / "train.xyz"
    write(data_path, Atoms("H"))

    initial_pot = tmp_path / "missing.yace"

    with pytest.raises(TrainerError, match="Initial potential not found"):
        trainer.train(data_path, initial_potential=initial_pot)
