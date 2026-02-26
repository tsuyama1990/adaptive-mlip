from pathlib import Path
from typing import Any

import yaml


# Centralized YAML validation
def safe_load_yaml(content: str) -> dict[str, Any]:
    """
    Safely loads YAML content with basic sanitization.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML content: {e}"
        raise ValueError(msg) from e

    if not isinstance(data, dict):
        msg = "YAML content must be a dictionary"
        raise TypeError(msg)
    return data

def load_yaml_file(filepath: Path) -> dict[str, Any]:
    """
    Safely loads YAML from a file.
    """
    if not filepath.exists():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)

    with filepath.open("r") as f:
        return safe_load_yaml(f.read())
