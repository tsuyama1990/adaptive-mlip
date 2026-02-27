from pathlib import Path

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.utils.io import load_yaml


def load_project_config(config_path: str | Path) -> PyAceConfig:
    """
    Loads and validates the project configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated PyAceConfig object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the configuration is invalid.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(config_path)
    config_dict = load_yaml(path)

    # Pydantic validation
    return PyAceConfig(**config_dict)
