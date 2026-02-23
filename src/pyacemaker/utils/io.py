from pathlib import Path
from typing import Any

import yaml

from pyacemaker.domain_models import PyAceConfig


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Loads a YAML file into a dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            msg = f"Error parsing YAML file: {e}"
            raise ValueError(msg) from e
        else:
            if not isinstance(data, dict):
                # Handle empty file or just scalar
                if data is None:
                    return {}
                msg = "YAML file must contain a dictionary"
                raise ValueError(msg)
            return data


def load_config(file_path: str | Path) -> PyAceConfig:
    """
    Loads a configuration file and validates it against the PyAceConfig schema.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        Validated PyAceConfig object.
    """
    data = load_yaml(file_path)
    # Pydantic will raise ValidationError if data is invalid
    return PyAceConfig(**data)
