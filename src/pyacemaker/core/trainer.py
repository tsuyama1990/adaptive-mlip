import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase.io import read, write

from pyacemaker.core.base import BaseTrainer
from pyacemaker.core.config_generator import PacemakerConfigGenerator
from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.utils.io import dump_yaml
from pyacemaker.utils.process import run_command


class PacemakerTrainer(BaseTrainer):
    """
    Pacemaker implementation of BaseTrainer.
    Wraps the 'pace_train' command.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.config_generator = PacemakerConfigGenerator(config)

    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:
        """
        Trains a potential using the provided training data file.

        This method wraps the external 'pace_train' command.
        It generates 'input.yaml' configuration for Pacemaker and executes the training.

        Args:
            training_data_path: Path to the file containing labelled structures.
                                Supported formats: .pckl, .xyz, .extxyz, .gzip.
            initial_potential: Optional path to an existing potential to fine-tune from.

        Returns:
            Path: The path to the generated potential file (e.g., potential.yace).

        Raises:
            TrainerError: If the training data file does not exist or format is invalid.
        """
        # Ensure pace_train is installed
        if not shutil.which("pace_train"):
            msg = "Executable 'pace_train' not found in PATH."
            raise TrainerError(msg)

        data_path = Path(training_data_path).resolve()
        self._validate_training_data(data_path)

        # Determine output directory (same as data file)
        output_dir = data_path.parent
        input_yaml_path = output_dir / "input.yaml"
        potential_path = output_dir / self.config.output_filename

        # Generate configuration
        pacemaker_config = self.config_generator.generate(str(data_path), str(potential_path))
        dump_yaml(pacemaker_config, input_yaml_path)

        # Run pace_train
        cmd = ["pace_train", str(input_yaml_path)]

        if initial_potential:
            initial_path = Path(initial_potential)
            if not initial_path.exists():
                msg = f"Initial potential not found: {initial_path}"
                raise TrainerError(msg)
            cmd.extend(["--initial_potential", str(initial_path)])

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            # Capture specific subprocess error
            msg = f"Training failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            # Catch other unexpected errors
            msg = f"Training failed unexpectedly: {e}"
            raise TrainerError(msg) from e

        if not potential_path.exists():
            msg = f"Potential file was not created at {potential_path}"
            raise TrainerError(msg)

        return potential_path

    def _validate_training_data(self, data_path: Path) -> None:
        """Validates existence and basic format of training data."""
        if not data_path.exists():
            msg = f"Training data not found: {data_path}"
            raise TrainerError(msg)

        if data_path.suffix not in {".pckl", ".xyz", ".extxyz", ".gzip"}:
            msg = f"Invalid training data format: {data_path.suffix}"
            raise TrainerError(msg)

        # Check for empty file
        if data_path.stat().st_size == 0:
            msg = f"Training data file is empty: {data_path}"
            raise TrainerError(msg)




class IncrementalTrainer(BaseTrainer):
    """
    A trainer wrapper that adds Incremental (Delta) Learning capabilities
    and manages a Replay Buffer to prevent catastrophic forgetting.
    """

    def __init__(self, base_trainer: BaseTrainer, replay_buffer_size: int = 500) -> None:
        self.base_trainer = base_trainer
        self.replay_buffer_size = replay_buffer_size

    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:
        """
        Trains a potential incrementally.

        1. Reads new structures from training_data_path.
        2. Appends to master history file (training_history.extxyz).
        3. Samples up to replay_buffer_size from history.
        4. Writes sampled structures to a temporary training set.
        5. Calls base_trainer.train with the temporary set and initial_potential.
        """
        data_path = Path(training_data_path).resolve()
        output_dir = data_path.parent
        history_path = output_dir / "training_history.extxyz"
        temp_train_path = output_dir / "training_set_temp.extxyz"

        # Read new structures
        new_structures = list(read(str(data_path), index=":"))

        # Read history if exists
        if history_path.exists():
            try:
                history_structures = list(read(str(history_path), index=":"))
            except Exception:
                history_structures = []
        else:
            history_structures = []

        # Append new structures to history and write back to disk
        all_structures = history_structures + new_structures
        write(str(history_path), all_structures, format="extxyz")

        # Sample for replay buffer
        if len(all_structures) > self.replay_buffer_size:
            sampled_structures = random.sample(all_structures, self.replay_buffer_size)
        else:
            sampled_structures = all_structures

        # Write out temp training set
        write(str(temp_train_path), sampled_structures, format="extxyz")

        # Train using base trainer
        return self.base_trainer.train(temp_train_path, initial_potential=initial_potential)
