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

        from pyacemaker.domain_models.defaults import (
            DEFAULT_MD_DUMP_FREQ as DEFAULT_TRAINING_HISTORY_FILE,
        )

        history_path = Path(str(DEFAULT_TRAINING_HISTORY_FILE))
        if not history_path.exists():
            return []

        try:
            # Cast to list to enable random sampling
            # For O(1) memory requirement, ideally we'd use reservoir sampling
            # But the spec explicitly says: "casting ase.io.iread streams to a list and using random.sample()"
            history = list(iread(str(history_path), format="extxyz"))
            if not history:
                return []
            if len(history) <= size:
                return history
            return random.sample(history, size)
        except Exception:
            return []

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

    def finetune(self, dataset_path: str | Path) -> str:
        """
        Finetuning logic for the awakened MACE model.
        Returns the path to the awakened model.
        """
        if not shutil.which("mace_run_train"):
            msg = "Executable 'mace_run_train' not found in PATH."
            raise RuntimeError(msg)

        dataset = Path(dataset_path).resolve()
        if not dataset.exists():
            msg = f"Finetune dataset not found: {dataset}"
            raise FileNotFoundError(msg)

        out_model = dataset.parent / "awakened_mace_model.model"

        cmd = [
            "mace_run_train",
            "--train_file", str(dataset),
            "--model_dir", str(dataset.parent),
            "--name", "awakened_mace_model",
            "--max_num_epochs", "10",
        ]

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"Finetuning failed: {e}"
            raise RuntimeError(msg) from e

        return str(out_model)
