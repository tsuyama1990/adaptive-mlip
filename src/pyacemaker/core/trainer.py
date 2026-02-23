import shutil
import subprocess
from pathlib import Path
from typing import Any

from pyacemaker.core.base import BaseTrainer
from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.utils.delta import get_lj_params
from pyacemaker.utils.io import detect_elements, dump_yaml
from pyacemaker.utils.process import run_command


class PacemakerTrainer(BaseTrainer):
    """
    Pacemaker implementation of BaseTrainer.
    Wraps the 'pace_train' command.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, training_data_path: str | Path) -> Any:
        """
        Trains a potential using the provided training data file.

        This method wraps the external 'pace_train' command.
        It generates 'input.yaml' configuration for Pacemaker and executes the training.

        Args:
            training_data_path: Path to the file containing labelled structures.
                                Supported formats: .pckl, .xyz, .extxyz, .gzip.

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
        pacemaker_config = self._generate_pacemaker_config(data_path, potential_path)
        dump_yaml(pacemaker_config, input_yaml_path)

        # Run pace_train
        cmd = ["pace_train", str(input_yaml_path)]
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


    def _generate_pacemaker_config(
        self, data_path: Path, output_path: Path
    ) -> dict[str, Any]:
        """Generates the dictionary for Pacemaker input.yaml."""
        elements = self._get_elements(data_path)
        pm_conf = self.config.pacemaker

        config_dict: dict[str, Any] = {
            "cutoff": self.config.cutoff_radius,
            "seed": self.config.seed,
            "data": {"filename": str(data_path)},
            "potential": self._get_potential_config(elements),
            "fit": self._get_fit_config(),
            "backend": {
                "evaluator": pm_conf.evaluator,
                "batch_size": self.config.batch_size,
                "display_step": pm_conf.display_step,
            },
        }

        if self.config.delta_learning:
            config_dict["base_potential"] = self._get_base_potential_config(elements)

        return config_dict

    def _get_elements(self, data_path: Path) -> list[str]:
        """Determine elements from config or data file."""
        from pyacemaker.domain_models.defaults import DEFAULT_MAX_FRAMES_ELEMENT_DETECTION

        if self.config.elements:
            return sorted(self.config.elements)

        try:
            return detect_elements(
                data_path, max_frames=DEFAULT_MAX_FRAMES_ELEMENT_DETECTION
            )
        except ValueError as e:
            msg = f"Could not detect elements from {data_path}. Please provide 'elements' in config or ensure data is valid: {e}"
            raise TrainerError(msg) from e

    def _get_potential_config(self, elements: list[str]) -> dict[str, Any]:
        """Generate potential section of config."""
        pm_conf = self.config.pacemaker
        return {
            "delta_spline_bins": pm_conf.delta_spline_bins,
            "elements": elements,
            "embeddings": {
                el: {
                    "ndensity": pm_conf.ndensity,
                    "npot": pm_conf.embedding_type,
                    "fs_parameters": pm_conf.fs_parameters,
                    "maxwell": True,
                }
                for el in elements
            },
            "bonds": {
                "N": self.config.max_basis_size,
                "max_deg": pm_conf.max_deg,
                "r0": pm_conf.r0,
                "rad_base": pm_conf.rad_base,
                "rad_parameters": pm_conf.rad_parameters,
            },
        }

    def _get_fit_config(self) -> dict[str, Any]:
        """Generate fit section of config."""
        pm_conf = self.config.pacemaker
        return {
            "loss": {
                "kappa": pm_conf.loss_kappa,
                "L1_coeffs": pm_conf.loss_l1_coeffs,
                "L2_coeffs": pm_conf.loss_l2_coeffs,
            },
            "optimizer": pm_conf.optimizer,
            "maxiter": self.config.max_iterations,
            "repulsion_sigma": pm_conf.repulsion_sigma,
        }

    def _get_base_potential_config(self, elements: list[str]) -> dict[str, Any]:
        """Generate base potential section for delta learning."""
        lj_params = {el: get_lj_params(el) for el in elements}
        return {
            "type": "LennardJones",
            "parameters": lj_params,
        }
