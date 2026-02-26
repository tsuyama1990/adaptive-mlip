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
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a dictionary")
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML content: {e}") from e

def load_yaml_file(filepath: Path) -> dict[str, Any]:
    """
    Safely loads YAML from a file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with filepath.open("r") as f:
        return safe_load_yaml(f.read())
