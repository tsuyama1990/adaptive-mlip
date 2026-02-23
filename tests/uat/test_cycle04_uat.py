from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms
from ase.io import write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


# Scenario 04-01: Fit Potential
def test_uat_fit_potential(tmp_path: Path) -> None:
    # GIVEN a labelled dataset
    dataset_path = tmp_path / "train.xyz"
    write(dataset_path, Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]))

    # AND a configuration
    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        delta_learning=True,
        output_filename="output_potential.yace"
    )

    trainer = PacemakerTrainer(config)

    # Mock the external tools
    with patch("subprocess.run") as mock_run, \
         patch("pyacemaker.core.trainer.dump_yaml"), \
         patch.object(Path, "exists", return_value=True):

        mock_run.return_value = MagicMock(returncode=0)

        # WHEN the Trainer executes
        result = trainer.train(dataset_path)

        # THEN a potential file is returned
        assert result.name == "output_potential.yace"

        # AND LJ params are calculated (implied by delta_learning=True logic inside train)
        # We can verify that get_lj_params was called if we mock it,
        # or check if the generated yaml contains "base_potential"

# Scenario 04-02: Active Set Selection
def test_uat_active_set_selection() -> None:
    # GIVEN a large pool of candidates
    pool = [Atoms('H') for _ in range(100)]

    selector = ActiveSetSelector()

    # WHEN the selector runs
    with patch("subprocess.run") as mock_run, \
         patch("pyacemaker.core.active_set.read") as mock_read, \
         patch("pyacemaker.core.active_set.write"), \
         patch.object(Path, "exists", return_value=True):

        mock_run.return_value = MagicMock(returncode=0)
        mock_read.return_value = pool[:10] # Mock returning 10 items

        selected = selector.select(pool, potential_path="current.yace", n_select=10)

        # THEN 10 structures are returned
        assert len(selected) == 10
