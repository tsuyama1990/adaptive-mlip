from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


def test_uat_fit_potential(tmp_path: Path) -> None:
    # GIVEN a labelled dataset
    dataset_path = tmp_path / "train.xyz"
    write(dataset_path, Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]))

    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        delta_learning=True,
        output_filename="output_potential.yace",
        elements=["H"],
    )
    trainer = PacemakerTrainer(config)

    # Use run_command patch to simulate success without real Pacemaker
    # And verify config file content
    with (
        patch("pyacemaker.core.trainer.run_command") as mock_run,
        patch("pyacemaker.core.trainer.dump_yaml") as mock_dump,
        patch("shutil.which", return_value="/usr/bin/pace_train"),
    ):  # Mock executable check
        # Simulate output file creation
        (tmp_path / "output_potential.yace").touch()

        result = trainer.train(dataset_path)

        assert result.name == "output_potential.yace"
        mock_run.assert_called_once()

        # Verify dumped config in UAT context
        args, _ = mock_dump.call_args
        config_dict = args[0]
        assert config_dict["potential"]["elements"] == ["H"]

        # Verify new advanced settings are passed correctly
        assert config_dict["potential"]["delta_spline_bins"] == 100
        assert config_dict["backend"]["evaluator"] == "tensorpot"
        assert config_dict["backend"]["display_step"] == 50


def test_uat_fit_potential_failure(tmp_path: Path) -> None:
    # GIVEN a labelled dataset
    dataset_path = tmp_path / "train.xyz"
    write(dataset_path, Atoms("H"))

    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        delta_learning=True,
        output_filename="output_potential.yace",
        elements=["H"],
    )
    trainer = PacemakerTrainer(config)

    # Simulate process failure
    import subprocess

    from pyacemaker.core.exceptions import TrainerError

    with (
        patch("pyacemaker.core.trainer.run_command") as mock_run,
        patch("shutil.which", return_value="/usr/bin/pace_train"),
    ):
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Simulated Crash")

        with pytest.raises(TrainerError, match="Training failed with exit code 1"):
            trainer.train(dataset_path)


def test_uat_active_set_selection(tmp_path: Path) -> None:
    pool = [Atoms("H") for _ in range(20)]
    pot_path = tmp_path / "current.yace"
    pot_path.touch()

    selector = ActiveSetSelector()

    # The actual implementation of ActiveSetSelector relies on `pace_active_set`
    # We must not mock the SUT logic itself, but we can mock the external process call
    # if it's strictly a driver/wrapper test. However, we should test pure logic.
    # We can mock `_write_structures` and `_read_structures` to avoid I/O entirely.

    with (
        patch.object(selector, "_write_candidates", return_value=20) as mock_write,
        patch.object(selector, "_execute_selection") as mock_execute,
        patch("pyacemaker.core.active_set.iread", return_value=iter(pool[:10])) as mock_read,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=100)),
    ):
        selected_iter = selector.select(pool, pot_path, n_select=10)

        # Test pure streaming consumption without materializing a list
        first = next(selected_iter)
        assert first is not None
        remaining = sum(1 for _ in selected_iter)
        assert remaining == 9

        mock_write.assert_called_once()
        mock_execute.assert_called_once()
        mock_read.assert_called_once()
