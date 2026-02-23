import subprocess
from pathlib import Path
from typing import Any

from ase.io import read

from pyacemaker.core.base import BaseTrainer
from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.utils.delta import get_lj_params
from pyacemaker.utils.io import dump_yaml


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
        if not data_path.exists():
            msg = f"Training data not found: {data_path}"
            raise TrainerError(msg)

        # Validate extension
        if data_path.suffix not in {".pckl", ".xyz", ".extxyz", ".gzip"}:
            msg = f"Invalid training data format: {data_path.suffix}"
            raise TrainerError(msg)

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
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,  # Security: explicit False
            )
        except subprocess.CalledProcessError as e:
            msg = f"Training failed (exit code {e.returncode}): {e.stderr}"
            raise TrainerError(msg) from e
        except FileNotFoundError as e:
            # Handle case where pace_train is not installed
            msg = "pace_train command not found. Ensure Pacemaker is installed."
            raise TrainerError(msg) from e

        if not potential_path.exists():
            # Sometimes pace_train might output to a different name if configured incorrectly
            # or if it crashed without non-zero exit code (rare).
            # We check if output_potential.yace exists as per config
            msg = f"Potential file was not created at {potential_path}"
            raise TrainerError(msg)

        return potential_path

    def _generate_pacemaker_config(
        self, data_path: Path, output_path: Path
    ) -> dict[str, Any]:
        """Generates the dictionary for Pacemaker input.yaml."""
        # Detect elements from the dataset
        # Strategy: Use elements from config if provided, otherwise detect from first frame.
        elements: list[str] = []
        if self.config.elements:
            elements = sorted(self.config.elements)
        else:
            try:
                # Use format='extxyz' if file extension allows, and index=0 to read only first frame
                fmt = "extxyz" if data_path.suffix == ".xyz" else None
                # Read single frame to detect elements.
                # Assuming homogeneous dataset where first frame contains all species or at least types
                # are consistent. If dataset is heterogeneous (e.g. pure A + pure B), this might miss elements.
                # But for MLIP training, usually structures contain all active species or we define them.
                # For safety, users SHOULD provide 'elements' in config for complex cases.
                atoms = read(data_path, index=0, format=fmt)
                if isinstance(atoms, list):
                    atoms = atoms[0]
                # get_chemical_symbols returns all symbols in structure.
                # set() gets unique ones.
                elements = sorted(set(atoms.get_chemical_symbols()))  # type: ignore[no-untyped-call]
            except Exception as e:
                msg = f"Could not detect elements from {data_path}. Please provide 'elements' in config or ensure data is valid: {e}"
                raise TrainerError(msg) from e

        config_dict: dict[str, Any] = {
            "cutoff": self.config.cutoff_radius,
            "seed": self.config.seed,
            "data": {"filename": str(data_path)},
            "potential": {
                "delta_spline_bins": 100,
                "elements": elements,
                "embeddings": {
                    el: {
                        "ndensity": 2,
                        "npot": "FinnisSinclair",
                        "fs_parameters": [1, 1, 1, 1.5],
                        "maxwell": True,
                    }
                    for el in elements
                },
                "bonds": {
                    "N": self.config.max_basis_size,
                    "max_deg": 6,
                    "r0": 1.5,
                    "rad_base": "Chebyshev",
                    "rad_parameters": [1.0],
                },
            },
            "fit": {
                "loss": {
                    "kappa": 0.3,
                    "L1_coeffs": 1e-8,
                    "L2_coeffs": 1e-8,
                },
                "optimizer": "BFGS",
                "maxiter": self.config.max_iterations,
                "repulsion_sigma": 0.05,
            },
            "backend": {
                "evaluator": "tensorpot",
                "batch_size": self.config.batch_size,
                "display_step": 50,
            },
        }

        # Handle Delta Learning (LJ Baseline)
        if self.config.delta_learning:
            lj_params = {}
            for el in elements:
                lj_params[el] = get_lj_params(el)

            config_dict["base_potential"] = {
                "type": "LennardJones",
                "parameters": lj_params
            }

        return config_dict
