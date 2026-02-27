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

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if not path.is_file():
        # Could be a directory
        raise FileNotFoundError(f"Configuration path is not a file: {path}")

    try:
        config_dict = load_yaml(path)
    except Exception as e:
        # Re-raise with user friendly message if it's a parsing error
        raise ValueError(f"Failed to parse configuration file {path}: {e}") from e

    if not isinstance(config_dict, dict):
        raise ValueError(f"Configuration file {path} must contain a dictionary")

    # Pydantic validation
    return PyAceConfig(**config_dict)
