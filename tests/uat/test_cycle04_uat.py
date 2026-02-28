from pathlib import Path
from typing import Any
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

    # Mocking run_command to simulate writing output
    def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
        # cmd is list
        out_idx = cmd.index("--output")
        out_path = Path(cmd[out_idx + 1])
        write(out_path, pool[:10], format="extxyz")
        return MagicMock()

    with patch("pyacemaker.core.active_set.run_command") as mock_run:
        mock_run.side_effect = side_effect

        from itertools import islice
        selected_iter = selector.select(pool, pot_path, n_select=10)
        selected = list(islice(selected_iter, 10))

        assert len(selected) == 10
