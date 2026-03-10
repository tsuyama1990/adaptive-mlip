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
        return []  # Mock replay buffer retrieval for now

    def incremental_train(
        self,
        new_data_path: str | Path,
        strategy_config: LoopStrategyConfig,
        initial_potential: str | Path | None = None,
    ) -> Any:
        """
        Mixes a replay buffer with the new active learning data and runs incremental delta learning.
        Streams data to pace_train using a named pipe to avoid O(N) memory/disk overhead.
        """
        import itertools
        import os
        import tempfile
        import threading

        from ase.io import iread, write

        replay_buffer = self.get_replay_buffer(strategy_config.replay_buffer_size)

        try:
            new_data_iter = iread(new_data_path, format="extxyz")
        except Exception:
            new_data_iter = iter([])

        combined_iter = itertools.chain(replay_buffer, new_data_iter)

        # Create a named pipe
        tmpdir = tempfile.mkdtemp()
        fifo_path = Path(tmpdir) / "stream.extxyz"
        os.mkfifo(fifo_path)

        # Write to the pipe in a background thread so we don't block
        def _writer() -> None:
            try:
                chunk_size = 100
                with fifo_path.open("w") as f_out:
                    while True:
                        chunk = list(itertools.islice(combined_iter, chunk_size))
                        if not chunk:
                            break
                        # Write the chunk directly to the open file object
                        write(f_out, chunk, format="extxyz")
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Error writing to pipe")
            # When the with block exits, f_out is closed, sending EOF to pace_train.

        writer_event = threading.Event()

        def _sync_writer() -> None:
            try:
                _writer()
            finally:
                writer_event.set()

        writer_thread = threading.Thread(target=_sync_writer, daemon=True)
        writer_thread.start()

        try:
            return self.train(fifo_path, initial_potential)
        finally:
            # If train failed before opening the pipe, the writer thread will be blocked forever.
            # We open it for reading non-blocking to unblock the writer thread and let it exit.
            if not writer_event.is_set():
                import contextlib
                with contextlib.suppress(OSError):
                    os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            writer_event.wait(timeout=5.0)
            # Cleanup pipe and temp dir
            try:
                fifo_path.unlink(missing_ok=True)
                Path(tmpdir).rmdir()
            except OSError:
                pass

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

        # Security: Schema validation using Pydantic instead of regex
        from pydantic import ValidationError

        from pyacemaker.domain_models.pacemaker import PacemakerInputSchema

        try:
            PacemakerInputSchema.model_validate(pacemaker_config)
        except ValidationError as e:
            msg = f"Generated Pacemaker config failed schema validation: {e}"
            raise TrainerError(msg) from e

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
        # Named pipes (FIFOs) have size 0, so skip the check for them
        if data_path.is_file() and data_path.stat().st_size == 0:
            msg = f"Training data file is empty: {data_path}"
            raise TrainerError(msg)


class FinetuneManager:
    """
    Manager to briefly train the final readout layers of the MACE foundation model.
    """

    def finetune(self, dataset_path: str | Path) -> str:
        """
        Finetunes the awakened MACE model using the provided dataset.
        Returns the path to the awakened model.
        """
        import os
        import shutil
        import subprocess

        from pyacemaker.utils.process import run_command

        mace_train_cmd = "mace_run_train"
        if not shutil.which(mace_train_cmd):
            msg = f"Executable '{mace_train_cmd}' not found in PATH."
            raise TrainerError(msg)

        output_model = "awakened_mace_model.model"

        # Read foundation model from env or use default
        foundation_model = os.environ.get("MACE_FOUNDATION_MODEL", "mace-mp-0-medium")

        cmd = [
            mace_train_cmd,
            "--name",
            "awakened_mace_model",
            "--train_file",
            str(dataset_path),
            "--foundation_model",
            foundation_model,
        ]

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"MACE Finetuning failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            msg = f"MACE Finetuning failed unexpectedly: {e}"
            raise TrainerError(msg) from e

        return output_model
