from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
        elements=["H"]
    )
    trainer = PacemakerTrainer(config)

    # Use run_command patch to simulate success without real Pacemaker
    with patch("pyacemaker.core.trainer.run_command") as mock_run, \
         patch("pyacemaker.core.trainer.dump_yaml"):

        # Simulate output file creation
        (tmp_path / "output_potential.yace").touch()

        result = trainer.train(dataset_path)

        assert result.name == "output_potential.yace"
        mock_run.assert_called_once()

def test_uat_active_set_selection(tmp_path: Path) -> None:
    pool = [Atoms('H') for _ in range(20)]
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

        selected_iter = selector.select(pool, pot_path, n_select=10)
        selected = list(selected_iter)

        assert len(selected) == 10
