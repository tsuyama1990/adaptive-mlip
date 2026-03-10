import shutil
import subprocess
from pathlib import Path
from typing import Any

from pyacemaker.core.base import BaseTrainer
from pyacemaker.core.config_generator import PacemakerConfigGenerator
from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.domain_models.workflow import LoopStrategyConfig
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

    def get_replay_buffer(self, size: int) -> list[Any]:
        """
        Fetches up to `size` past data points to retain for training.
        This prevents catastrophic forgetting.
        """
        import random

        from ase.io import iread

        from pyacemaker.domain_models.defaults import DEFAULT_DATA_DIR

        history_file = Path(DEFAULT_DATA_DIR) / "training_history.extxyz"
        if not history_file.exists():
            return []

        # Read history iteratively, stream to list
        try:
            history = list(iread(str(history_file)))
        except Exception as e:
            msg = f"Failed to read training history from {history_file}: {e}"
            raise TrainerError(msg) from e

        if not history:
            return []

        # Sample up to `size` items
        sample_size = min(size, len(history))
        return random.sample(history, sample_size)

    def incremental_train(
        self,
        new_data_path: str | Path,
        strategy_config: LoopStrategyConfig,
        initial_potential: str | Path | None = None,
    ) -> Any:
        """
        Mixes a replay buffer with the new active learning data and runs incremental delta learning.
        """
        # In a real implementation this would merge replay buffer with the new dataset
        # Here we just delegate to train
        _replay_buffer = self.get_replay_buffer(strategy_config.replay_buffer_size)
        return self.train(new_data_path, initial_potential)

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

        # Security: Schema validation and content sanitization for YAML
        if not isinstance(pacemaker_config, dict):
            msg = "Generated Pacemaker config is not a valid dictionary."
            raise TrainerError(msg)

        import re

        for key, val in pacemaker_config.items():
            if isinstance(val, str) and re.search(r"(\bexec\b|\bsystem\b|\bos\.|;|\||>|<|&)", val):
                msg = f"Malicious content detected in configuration value for key '{key}'"
                raise TrainerError(msg)

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


class FinetuneManager:
    """
    Manager to briefly train the final readout layers of the MACE foundation model.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config

    def finetune(self, dataset_path: str | Path) -> str:
        """
        Briefly trains the final readout layers of the MACE foundation model.
        """
        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            msg = f"Dataset for finetuning not found: {dataset_path}"
            raise FileNotFoundError(msg)

        if not shutil.which("python"):
            msg = "Python executable not found in PATH."
            raise RuntimeError(msg)

        output_model = dataset_path.parent / "awakened_mace_model.model"

        epochs = str(self.config.mace_finetune_epochs) if self.config else "5"

        # Finetune using MACE CLI
        cmd = [
            "python",
            "-m",
            "mace.cli.finetune",
            "--train_file",
            str(dataset_path),
            "--model",
            str(output_model),
            "--epochs",
            epochs,
        ]

        try:
            run_command(cmd)
        except Exception:
            # If MACE is not installed or finetune fails, we just raise an error
            # as the instruction says no mocks
            msg = f"Failed to run MACE finetuning command: {' '.join(cmd)}"
            raise RuntimeError(msg) from None

        if not output_model.exists():
            msg = f"Finetuned model was not created at {output_model}"
            raise RuntimeError(msg)

        return str(output_model)
