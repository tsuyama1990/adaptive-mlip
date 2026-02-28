from typing import Any

from pyacemaker.core.exceptions import TrainerError
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.utils.delta import get_lj_params
from pyacemaker.utils.io import detect_elements


class PacemakerConfigGenerator:
    """
    Generates Pacemaker configuration dictionaries.
    Extracted from PacemakerTrainer for Single Responsibility Principle.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def generate(self, data_path: str, output_path: str) -> dict[str, Any]:
        """
        Generates the configuration dictionary.

        Args:
            data_path: Path to the training data.
            output_path: Path to the output potential file.

        Returns:
            Configuration dictionary ready for dumping to YAML.
        """
        elements = self._get_elements(data_path)

        config_dict: dict[str, Any] = {
            "cutoff": self.config.cutoff_radius,
            "seed": self.config.seed,
            "data": {"filename": data_path},
            "potential": self._get_potential_config(elements),
            "fit": self._get_fit_config(),
            "backend": self._get_backend_config(),
        }

        if self.config.delta_learning:
            config_dict["base_potential"] = self._get_base_potential_config(elements)

        return config_dict

    def _get_elements(self, data_path: str) -> list[str]:
        """Determine elements from config or data file."""
        from pathlib import Path

        from pyacemaker.domain_models.defaults import DEFAULT_MAX_FRAMES_ELEMENT_DETECTION

        if self.config.elements:
            return sorted(self.config.elements)

        try:
            return detect_elements(Path(data_path), max_frames=DEFAULT_MAX_FRAMES_ELEMENT_DETECTION)
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

    def _get_backend_config(self) -> dict[str, Any]:
        """Generate backend section of config."""
        pm_conf = self.config.pacemaker
        return {
            "evaluator": pm_conf.evaluator,
            "batch_size": self.config.batch_size,
            "display_step": pm_conf.display_step,
        }

    def _get_base_potential_config(self, elements: list[str]) -> dict[str, Any]:
        """Generate base potential section for delta learning."""
        lj_params = {el: get_lj_params(el) for el in elements}
        return {
            "type": "LennardJones",
            "parameters": lj_params,
        }
