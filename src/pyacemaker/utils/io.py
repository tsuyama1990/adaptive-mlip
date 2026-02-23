from pathlib import Path
from typing import Any

import yaml

from pyacemaker.constants import (
    ERR_CONFIG_NOT_FOUND,
    ERR_PATH_NOT_FILE,
    ERR_PATH_TRAVERSAL,
    ERR_YAML_NOT_DICT,
    ERR_YAML_PARSE,
)
from pyacemaker.domain_models import PyAceConfig


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Loads a YAML file into a dictionary with path safety checks.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If path is invalid or attempts traversal.
        yaml.YAMLError: If the YAML is invalid.
    """
    path = Path(file_path).resolve()
    base_dir = Path.cwd().resolve()

    # Path Sanitization: Ensure path doesn't traverse outside allowed scope (CWD)
    if not path.is_relative_to(base_dir):
        msg = ERR_PATH_TRAVERSAL.format(path=path, base=base_dir)
        raise ValueError(msg)

    if not path.exists():
        msg = ERR_CONFIG_NOT_FOUND.format(path=path)
        raise FileNotFoundError(msg)

    # Ensure it's a file, not a directory
    if not path.is_file():
        msg = ERR_PATH_NOT_FILE.format(path=path)
        raise ValueError(msg)

    with path.open("r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            msg = ERR_YAML_PARSE.format(error=e)
            raise ValueError(msg) from e
        else:
            if not isinstance(data, dict):
                 # Handle empty file or just scalar
                 if data is None:
                     return {}
                 raise ValueError(ERR_YAML_NOT_DICT)
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
