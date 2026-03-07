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
        if not isinstance(use_mock, bool):
            msg = "use_mock must be a boolean"
            raise TypeError(msg)
        self.use_mock = use_mock

    def finetune(self, dft_data_path: Path) -> None:
        from pyacemaker.utils.path import validate_path_safe
        dft_data_path = validate_path_safe(dft_data_path)
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
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Path | None:
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
        data_path = Path(training_data_path)
        initial_potential_path = Path(initial_potential) if initial_potential else None

        # Ensure pace_train is installed
        import os

        pace_train_exe = os.environ.get("PACE_TRAIN_EXECUTABLE", "pace_train")
        if not shutil.which(pace_train_exe):
            msg = f"Executable '{pace_train_exe}' not found in PATH."
            raise TrainerError(msg)

        self._validate_training_data(data_path)

        # Determine output directory (same as data file)
        from pyacemaker.utils.path import validate_path_safe

        output_dir = validate_path_safe(data_path.parent)
        input_yaml_path = output_dir / "input.yaml"
        potential_path = validate_path_safe(output_dir / self.config.output_filename)

        # Generate configuration
        pacemaker_config = self.config_generator.generate(str(data_path), str(potential_path))
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

    def incremental_train(
        self, new_data_path: str | Path, replay_buffer_path: str | Path | None = None, replay_buffer_size: int = 500
    ) -> Path | None:
        """
        Performs Delta Learning using an active replay buffer to prevent catastrophic forgetting.
        Uses streaming data mixing to satisfy O(1) memory requirements.
        """
        data_path = Path(new_data_path)
        replay_path = Path(replay_buffer_path) if replay_buffer_path else None

        self._validate_training_data(data_path)

        mixed_data_path = data_path

        if replay_path and replay_path.exists():
            import itertools
            import logging

            import numpy as np
            from ase.io import iread, write

            mixed_data_path = data_path.with_name(f"{data_path.stem}_mixed{data_path.suffix}")

            # Read new data using iterator
            new_data_iter = iread(str(data_path), index=":")
            # Write new data out to the mixed file (overwrite mode initially)
            for chunk in itertools.batched(new_data_iter, 50): # type: ignore[attr-defined]
                write(str(mixed_data_path), chunk, append=True)

            # Sample replay buffer using memory-efficient reservoir sampling or bounded generator
            # For simplicity without materialization: read all, randomly decide to yield, but limit to size.
            replay_count = 0
            # To actually sample uniformly from an iterator, we can use reservoir sampling
            reservoir = []
            for i, atoms in enumerate(iread(str(replay_path), index=":")):
                if len(reservoir) < replay_buffer_size:
                    reservoir.append(atoms)
                else:
                    j = np.random.randint(0, i + 1)
                    if j < replay_buffer_size:
                        reservoir[j] = atoms
                replay_count += 1

            # Write sampled replay to the mixed file
            for chunk in itertools.batched(reservoir, 50): # type: ignore[attr-defined]
                write(str(mixed_data_path), chunk, append=True)

            logging.getLogger(__name__).info(f"Mixed new structures with {min(replay_count, replay_buffer_size)} replay structures.")

        # Now fallback to train using the previous potential (or base potential) as initial
        return self.train(mixed_data_path)

    def get_replay_buffer(self) -> Path | None:
        """
        Returns the path to the current replay buffer.
        """
        return None

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
