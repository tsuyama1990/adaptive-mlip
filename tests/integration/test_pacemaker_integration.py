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


def test_pacemaker_integration_full_flow(tmp_path: Path) -> None:
    """
    Integration test for PacemakerTrainer.
    Verifies:
    1. Input file reading (element detection).
    2. Config generation (using defaults).
    3. Command execution (mocked).
    4. Output file handling.
    """
    # 1. Setup Data
    data_path = tmp_path / "training_data.xyz"
    # Write a few frames to test element detection
    atoms = [Atoms("H2O"), Atoms("H2O")]
    write(data_path, atoms)

    output_pot_path = tmp_path / "output.yace"

    # 2. Config
    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=4.0,
        max_basis_size=300,
        output_filename="output.yace",
        # No elements specified, relies on detection
    )

    trainer = PacemakerTrainer(config)

    # 3. Mock Execution Environment
    # We mock:
    # - shutil.which (to simulate pace_train exists)
    # - subprocess.run (to simulate successful execution)
    with patch("shutil.which", return_value="/usr/bin/pace_train"), \
         patch("pyacemaker.core.trainer.run_command") as mock_run:

        # Simulate output file creation by the external process
        def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
            # cmd[0] is 'pace_train', cmd[1] is input.yaml path
            input_yaml = Path(cmd[1])
            assert input_yaml.exists()

            # Verify generated YAML content
            with input_yaml.open() as f:
                data = yaml.safe_load(f)
                assert data["potential"]["elements"] == ["H", "O"]
                assert data["cutoff"] == 4.0

            # Create the expected output file
            output_pot_path.touch()
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # 4. Execute
        result = trainer.train(data_path)

        # 5. Verify
        assert result == output_pot_path
        assert result.exists()
        mock_run.assert_called_once()

def test_pacemaker_integration_failure_handling(tmp_path: Path) -> None:
    """Test trainer failure handling."""
    data_path = tmp_path / "data.xyz"
    write(data_path, Atoms("He"))

    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=3.0,
        max_basis_size=100
    )
    trainer = PacemakerTrainer(config)

    from pyacemaker.core.exceptions import TrainerError

    with patch("shutil.which", return_value="/bin/true"), \
         patch("pyacemaker.core.trainer.run_command") as mock_run:

        # Simulate process failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

        with pytest.raises(TrainerError, match="Training failed with exit code 1"):
            trainer.train(data_path)
