from pathlib import Path
from typing import Any

from ase.io import iread

from pyacemaker.core.base import BaseTrainer
from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.utils.delta import get_lj_params
from pyacemaker.utils.io import dump_yaml
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
        data_path = Path(training_data_path)
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
        except Exception as e:
            msg = f"Training failed: {e}"
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
        elements: list[str] = []
        if self.config.elements:
            elements = sorted(self.config.elements)
        else:
            try:
                # Scan first few frames to detect elements
                # Reading just one frame might be insufficient for mixed datasets.
                # Scanning 10 frames is a reasonable compromise.
                elements_set = set()
                fmt = "extxyz" if data_path.suffix == ".xyz" else None
                # Use iread for streaming
                # iread format requires str, not None. But ASE handles None = autodetect.
                # To satisfy mypy, we cast or default. 'extxyz' is safest if extension matches.
                # If fmt is None, iread uses filename.
                read_fmt = fmt if fmt else ""
                for i, atoms in enumerate(iread(data_path, index=":", format=read_fmt)):
                    # atoms from iread is Atoms object
                    elements_set.update(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
                    if i >= 10:  # Stop after 10 frames
                        break
                elements = sorted(elements_set)

                if not elements:
                     msg = "No elements detected in training data (file might be effectively empty)."
                     raise TrainerError(msg)

            except Exception as e:
                msg = f"Could not detect elements from {data_path}. Please provide 'elements' in config or ensure data is valid: {e}"
                raise TrainerError(msg) from e

        # Use PacemakerConfig from TrainingConfig
        pm_conf = self.config.pacemaker

        config_dict: dict[str, Any] = {
            "cutoff": self.config.cutoff_radius,
            "seed": self.config.seed,
            "data": {"filename": str(data_path)},
            "potential": {
                "delta_spline_bins": 100,
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
            },
            "fit": {
                "loss": {
                    "kappa": pm_conf.loss_kappa,
                    "L1_coeffs": pm_conf.loss_l1_coeffs,
                    "L2_coeffs": pm_conf.loss_l2_coeffs,
                },
                "optimizer": pm_conf.optimizer,
                "maxiter": self.config.max_iterations,
                "repulsion_sigma": pm_conf.repulsion_sigma,
            },
            "backend": {
                "evaluator": "tensorpot",
                "batch_size": self.config.batch_size,
                "display_step": 50,
            },
        }

        if self.config.delta_learning:
            lj_params = {}
            for el in elements:
                lj_params[el] = get_lj_params(el)

            config_dict["base_potential"] = {
                "type": "LennardJones",
                "parameters": lj_params
            }

        return config_dict
