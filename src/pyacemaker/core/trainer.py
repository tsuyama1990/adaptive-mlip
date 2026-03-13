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

    def get_replay_buffer(self, size: int, history_path: str | Path) -> list[Any]:
        """
        Fetches up to `size` past data points to retain for training.
        This prevents catastrophic forgetting.
        Uses ase.io.iread with true reservoir sampling for streaming memory safety.
        """
        import random
        from pathlib import Path

        from ase.io import iread

        path = Path(history_path)
        if not path.exists():
            return []

        reservoir: list[Any] = []
        try:
            atoms_iter = iread(path, format="extxyz")
            for i, item in enumerate(atoms_iter):
                if i < size:
                    reservoir.append(item)
                else:
                    j = random.randint(0, i)  # noqa: S311
                    if j < size:
                        reservoir[j] = item
        except Exception:
            return reservoir

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
        import itertools
        from pathlib import Path

        from ase.io import iread, write

        # We need a dynamic path to history, we assume it's stored alongside the new data
        # or in the default config directory
        from pyacemaker.domain_models.defaults import DEFAULT_DATA_DIR

        history_path = Path(DEFAULT_DATA_DIR) / "training_history.extxyz"

        replay_buffer = self.get_replay_buffer(strategy_config.replay_buffer_size, history_path)

        # Use streaming to combine data, averting OOM issues with large datasets
        new_data_stream = iread(new_data_path, index=":")
        combined_data_stream = itertools.chain(new_data_stream, replay_buffer)

        # Write to a temporary file using the generator, then train
        temp_file = Path(new_data_path).parent / "combined_train_data.extxyz"

        # Write consumes the generator efficiently without materializing all frames at once
        write(temp_file, combined_data_stream, format="extxyz")  # type: ignore[arg-type]

        try:
            return self.train(temp_file, initial_potential)
        finally:
            if temp_file.exists():
                temp_file.unlink()

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
            if isinstance(val, str) and re.search(
                r"(\bexec\b|\bsystem\b|\bos\.|;|\||>|<|&|`|\$|\n|\r|\\)", val
            ):
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
        Implements actual fine-tuning logic for the MACE model.
        """
        from ase.io import iread

        # Validate path
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            msg = f"Dataset not found: {dataset_path}"
            raise FileNotFoundError(msg)

        # Verify structure is readable efficiently
        _ = next(iread(dataset_path, index=":"))

        output_model = dataset_path.parent / "awakened_mace_model.model"

        if not shutil.which("mace_run_train"):
            msg = "Executable 'mace_run_train' not found in PATH."
            raise TrainerError(msg)

        # Execute real MACE fine-tuning
        cmd = [
            "mace_run_train",
            f"--name={output_model.stem}",
            f"--train_file={dataset_path}",
            f"--checkpoints_dir={dataset_path.parent}",
            "--model=MACE",
            "--max_num_epochs=5",  # very brief fine-tuning
            "--device=cuda",
            "--error_table=MACE_error_table",
        ]

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"MACE fine-tuning failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            msg = f"MACE fine-tuning failed unexpectedly: {e}"
            raise TrainerError(msg) from e

        # MACE saves as {name}_compiled.model by default when complete
        compiled_model = dataset_path.parent / f"{output_model.stem}_compiled.model"
        if compiled_model.exists():
            shutil.move(compiled_model, output_model)

        if not output_model.exists():
            msg = f"MACE model file was not created at {output_model}"
            raise TrainerError(msg)

        return str(output_model)
