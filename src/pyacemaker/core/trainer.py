import shutil
import subprocess
from pathlib import Path

from pyacemaker.core.base import BaseTrainer
from pyacemaker.core.config_generator import PacemakerConfigGenerator
from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.utils.io import dump_yaml
from pyacemaker.utils.process import run_command


class FinetuneManager:
    """
    Mocked manager for fine-tuning the MACE foundational model.
    Real implementation would load the PyTorch MACE model and run a short
    fine-tuning loop on the newly acquired DFT data.
    """
    def __init__(self, use_mock: bool = False) -> None:
        self.use_mock = use_mock

    def finetune(self, dft_data_path: Path) -> None:
        if self.use_mock:
            import logging
            logging.getLogger(__name__).info(f"Mock MACE fine-tuning completed using {dft_data_path}")
        else:
            msg = "MACE model is not installed or available for fine-tuning."
            raise RuntimeError(msg)


class PacemakerTrainer(BaseTrainer):
    """
    Pacemaker implementation of BaseTrainer.
    Wraps the 'pace_train' command.
    """

    def __init__(self, config: TrainingConfig) -> None:
        if config is None:
            msg = "config cannot be None"
            raise ValueError(msg)
        self.config = TrainingConfig.model_validate(config)
        self.config_generator = PacemakerConfigGenerator(config)

    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None, replay_buffer_path: Path | None = None, replay_buffer_size: int = 500
    ) -> Path | None:
        from pyacemaker.utils.path import validate_path_safe

        # Validate all file paths
        data_path = validate_path_safe(Path(training_data_path))
        if initial_potential:
            initial_potential_path = validate_path_safe(Path(initial_potential))
        else:
            initial_potential_path = None

        if replay_buffer_path:
            replay_buffer_path_safe = validate_path_safe(Path(replay_buffer_path))
        else:
            replay_buffer_path_safe = None
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
        import os

        pace_train_exe = os.environ.get("PACE_TRAIN_EXECUTABLE", "pace_train")
        if not shutil.which(pace_train_exe):
            msg = f"Executable '{pace_train_exe}' not found in PATH."
            raise TrainerError(msg)

        self._validate_training_data(data_path)

        # Mix replay buffer if provided
        final_data_path = data_path
        if replay_buffer_path_safe and replay_buffer_path_safe.exists():
            import numpy as np
            from ase.io import read, write

            # Load new data
            new_data = list(read(str(data_path), index=":"))

            # Load and sample from replay buffer
            replay_data = list(read(str(replay_buffer_path_safe), index=":"))
            if len(replay_data) > replay_buffer_size:
                rng = np.random.default_rng()
                indices = rng.choice(len(replay_data), size=replay_buffer_size, replace=False)
                sampled_replay = [replay_data[i] for i in indices]
            else:
                sampled_replay = replay_data

            mixed_data = new_data + sampled_replay

            # Create a new mixed data file
            mixed_data_path = data_path.with_name(f"{data_path.stem}_mixed{data_path.suffix}")
            write(str(mixed_data_path), mixed_data)
            final_data_path = mixed_data_path

            import logging
            logging.getLogger(__name__).info(f"Mixed {len(new_data)} new structures with {len(sampled_replay)} replay structures.")

        # Determine output directory (same as data file)
        from pyacemaker.utils.path import validate_path_safe

        output_dir = validate_path_safe(final_data_path.parent)
        input_yaml_path = output_dir / "input.yaml"
        potential_path = validate_path_safe(output_dir / self.config.output_filename)

        # Generate configuration
        pacemaker_config = self.config_generator.generate(str(final_data_path), str(potential_path))
        dump_yaml(pacemaker_config, input_yaml_path)

        # Run pace_train
        cmd = [pace_train_exe, str(input_yaml_path)]

        if initial_potential_path:
            if not initial_potential_path.exists():
                msg = f"Initial potential not found: {initial_potential_path}"
                raise TrainerError(msg)
            cmd.extend(["--initial_potential", str(initial_potential_path)])

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
            import logging

            logging.getLogger(__name__).error(f"Potential file was not created at {potential_path}")
            return None

        return potential_path

    @staticmethod
    def _validate_training_data(data_path: Path) -> None:
        """Validates existence and basic format of training data."""
        if not data_path.exists():
            msg = f"Training data not found: {data_path}"
            raise TrainerError(msg)

        from pyacemaker.domain_models.defaults import SUPPORTED_TRAINING_FORMATS

        if data_path.suffix not in SUPPORTED_TRAINING_FORMATS:
            msg = f"Invalid training data format: {data_path.suffix}"
            raise TrainerError(msg)

        # Check for empty file
        if data_path.stat().st_size == 0:
            msg = f"Training data file is empty: {data_path}"
            raise TrainerError(msg)
