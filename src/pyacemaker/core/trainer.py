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
        This prevents catastrophic forgetting using reservoir sampling for O(1) memory overhead.
        """

        from ase.io import iread

        history_file = Path("training_history.extxyz")
        if not history_file.exists():
            return []

        try:
            stream = iread(str(history_file), format="extxyz")
            reservoir = []

            import secrets
            for i, frame in enumerate(stream):
                if i < size:
                    reservoir.append(frame)
                else:
                    j = secrets.randbelow(i + 1)
                    if j < size:
                        reservoir[j] = frame

        except Exception:
            return []
        else:
            return reservoir

    def incremental_train(
        self,
        new_data_path: str | Path,
        strategy_config: LoopStrategyConfig,
        initial_potential: str | Path | None = None,
    ) -> Any:
        """
        Mixes a replay buffer with the new active learning data and runs incremental delta learning.
        """
        from ase.io import read, write

        replay_buffer = self.get_replay_buffer(strategy_config.replay_buffer_size)

        # Merge replay buffer into the new data
        if replay_buffer:
            try:
                # Read new data, append replay, write back safely via temp file
                new_data = read(str(new_data_path), index=":", format="extxyz")
                if not isinstance(new_data, list):
                    new_data = [new_data]

                combined_data = new_data + replay_buffer

                temp_path = Path(str(new_data_path) + ".tmp")
                write(str(temp_path), combined_data, format="extxyz")
                shutil.move(str(temp_path), str(new_data_path))
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to merge replay buffer: {e}")
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
                # Proceed with just new data if merge fails

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

        if not isinstance(pacemaker_config, dict):
            msg = "Generated Pacemaker config is not a valid dictionary."
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
        """Validates existence, basic format, and integrity of training data."""
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

        def _raise_empty() -> None:
            msg = f"First frame of training data is empty: {data_path}"
            raise TrainerError(msg)

        # Verify content parses cleanly
        try:
            from ase.io import iread
            first_frame = next(iread(str(data_path)))
            if not len(first_frame):
                _raise_empty()
        except TrainerError:
            raise
        except Exception as e:
            msg = f"Training data failed integrity parsing check: {e}"
            raise TrainerError(msg) from e


class FinetuneManager:
    """
    Manager to briefly train the final readout layers of the MACE foundation model.
    """

    def finetune(self, dataset_path: str | Path) -> str:
        """
        Finetunes the MACE model with new ground truth DFT data.
        Returns the path to the awakened model.
        """
        from ase.io import read

        data_file = Path(dataset_path)
        if not data_file.exists():
            msg = f"Finetune dataset not found: {data_file}"
            raise TrainerError(msg)

        try:
            frames = read(str(data_file), index=":")
        except Exception as e:
            msg = f"Failed to read dataset: {e}"
            raise TrainerError(msg) from e

        if not frames:
            msg = "No frames to finetune."
            raise TrainerError(msg)

        # Real implementation, call mace_run_train to finetune
        if not shutil.which("mace_run_train"):
            msg = "Executable 'mace_run_train' not found in PATH."
            raise TrainerError(msg)

        cmd = [
            "mace_run_train",
            "--train_file", str(data_file),
            "--valid_fraction", "0.0",
            "--E0s", "average",
            "--model", "MACE",
            "--num_interactions", "2",
            "--max_num_epochs", "10",
            "--start_swa", "5",
            "--scheduler_patience", "5",
            "--patience", "10",
            "--eval_interval", "1",
            "--loss", "forces_only",
            "--device", "cuda",
            "--name", "awakened_mace_model",
            "--train_dir", str(data_file.parent)
        ]

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"MACE Finetuning failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            msg = f"MACE Finetuning failed unexpectedly: {e}"
            raise TrainerError(msg) from e

        # Find the output model
        mace_model_path = data_file.parent / "awakened_mace_model.model"
        if not mace_model_path.exists():
            msg = f"MACE Finetuning did not produce a model at {mace_model_path}"
            raise TrainerError(msg)

        return str(mace_model_path)
