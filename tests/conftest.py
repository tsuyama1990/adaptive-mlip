from typing import Any

import pytest

from pyacemaker.domain_models import PyAceConfig


# Reusable test utilities
def create_test_config_dict(**overrides: Any) -> dict[str, Any]:
    """Creates a dictionary for a valid PyAceConfig."""
    defaults: dict[str, Any] = {
        "project_name": "TestProject",
        "structure": {"elements": ["Fe", "Pt"], "supercell_size": [1, 1, 1]},
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 500.0,
            "pseudopotentials": {"Fe": "Fe.UPF", "Pt": "Pt.UPF"},
        },
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 500},
        "md": {"temperature": 1000.0, "pressure": 0.0, "timestep": 0.001, "n_steps": 1000},
        "workflow": {"max_iterations": 10},
        "logging": {"log_file": "test.log"},
    }

    # Simple merge
    for section, values in overrides.items():
        if section in defaults and isinstance(defaults[section], dict):
            defaults[section].update(values)
        else:
            defaults[section] = values

    return defaults

@pytest.fixture
def mock_config_obj() -> PyAceConfig:
    return PyAceConfig.model_validate(create_test_config_dict())
