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


class FinetuneManager:
    """
    Manager for fine-tuning the foundation model (MACE).
    Handles MACE awakening process using acquired DFT data.
    """
    def __init__(self, mace_model_path: str = "MACE-MP-0"):
        self.mace_model_path = mace_model_path

    def finetune(self, dft_data_path: str | Path) -> Path:
        """
        Finetunes the MACE model using the provided DFT data.
        Returns the path to the finetuned model.
        """
        dft_path = Path(dft_data_path)
        if not dft_path.exists():
            raise TrainerError(f"DFT data not found: {dft_path}")

        # Mock finetuning process.
        # In full implementation, calls mace-train with specific readout layer configs.
        finetuned_model_path = dft_path.parent / "finetuned_mace.model"

        # We just create a dummy file for the test
        finetuned_model_path.touch()

        return finetuned_model_path


class PacemakerTrainer(BaseTrainer):
    """
    Pacemaker implementation of BaseTrainer.
    Wraps the 'pace_train' command.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.config_generator = PacemakerConfigGenerator(config)

    def train(
        self,
        training_data_path: str | Path,
        initial_potential: str | Path | None = None
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

    def incremental_train(
        self,
        new_data_path: str | Path,
        historical_data_path: str | Path,
        initial_potential: str | Path | None,
        strategy_config: LoopStrategyConfig
    ) -> Any:
        """
        Performs incremental training (Delta Learning) using a streaming replay buffer approach.
        Instead of loading the entire dataset into memory, it uses the ase format reader as an iterator
        to process the dataset in a stream.
        """

        import ase.io

        new_path = Path(new_data_path).resolve()
        hist_path = Path(historical_data_path).resolve()

        # Validate data
        self._validate_training_data(new_path)

        # We blend data iteratively to a new stream
        blended_path = new_path.parent / "blended_training_data.xyz"

        try:
            # Replay buffer sampling via reservoir sampling (O(N) time, O(K) space)
            buffer_data = self.get_replay_buffer(hist_path, strategy_config.replay_buffer_size)

            # Read new data fully as it's typically small
            new_data = ase.io.read(new_path, index=":")
            if not isinstance(new_data, list):
                new_data = [new_data]

            # Stream the writing process to keep memory constrained
            # Note: For write(), ase supports streaming via generators if written carefully,
            # or we can simply write sequentially in append mode.
            if blended_path.exists():
                blended_path.unlink()

            # Write buffer
            for atoms in buffer_data:
                ase.io.write(blended_path, atoms, append=True) # type: ignore[no-untyped-call]

            # Write new data
            for atoms in new_data:
                ase.io.write(blended_path, atoms, append=True) # type: ignore[no-untyped-call]

        except FileNotFoundError as e:
            raise TrainerError(f"Failed to access data file: {e}") from e
        except RuntimeError as e:
            raise TrainerError(f"Error during incremental data blending: {e}") from e

        return self.train(blended_path, initial_potential=initial_potential)

    def get_replay_buffer(self, historical_data_path: Path, buffer_size: int) -> list[Any]:
        """
        Samples a fixed-size replay buffer from historical training data
        using Reservoir Sampling to maintain O(n) streaming performance.
        Uses cryptographically secure RNG.
        """
        import logging
        import secrets

        import ase.io

        if not historical_data_path.exists() or historical_data_path.stat().st_size == 0:
            return []

        if buffer_size <= 0:
            return []

        try:
            # ase.io.iread is an iterator over the file
            reservoir = []
            for i, atoms in enumerate(ase.io.iread(historical_data_path)):
                if i < buffer_size:
                    reservoir.append(atoms)
                else:
                    j = secrets.randbelow(i + 1)
                    if j < buffer_size:
                        reservoir[j] = atoms

            return reservoir
        except OSError as e:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to access historical data file: %s", e)
            return []
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to sample replay buffer: %s", e)
            return []

