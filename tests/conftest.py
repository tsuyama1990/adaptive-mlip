from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError

from pyacemaker.domain_models import DFTConfig


@pytest.fixture
def mock_dft_config() -> DFTConfig:
    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF", "O": "O.UPF", "Fe": "Fe.UPF"},
    )


class MockCalculator(Calculator):
    """
    Mock ASE calculator for testing purposes.
    Can simulate failures and setup errors.
    """

    def __init__(self, fail_count: int = 0, setup_error: bool = False) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.implemented_properties = ["energy", "forces", "stress"]
        self.fail_count = fail_count
        self.setup_error = setup_error
        self.attempts = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        self.attempts += 1

        if self.setup_error:
            msg = "Setup failed"
            raise CalculatorSetupError(msg)

        if self.attempts <= self.fail_count:
            # Simulate SCF failure
            msg = "Convergence not achieved"
            raise RuntimeError(msg)

        self.results = {
            "energy": -13.6,
            "forces": np.array([[0.0, 0.0, 0.0]] * (len(atoms) if atoms else 1)),
            "stress": np.array([0.0] * 6),
        }

def create_test_config_dict(**overrides: Any) -> dict[str, Any]:
    """Helper to create a valid config dictionary with optional overrides."""
    base_config = {
        "project_name": "TestProject",
        "structure": {
            "elements": ["Fe"],
            "supercell_size": [1, 1, 1],
            "adaptive_ratio": 0.5,
            "defect_density": 0.01,
            "strain_range": 0.05
        },
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 500.0,
            "pseudopotentials": {"Fe": "Fe.UPF"},
            "mixing_beta": 0.7,
            "smearing_type": "mv",
            "smearing_width": 0.1,
            "diagonalization": "david"
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 5.0,
            "max_basis_size": 500,
            "delta_learning": True,
            "active_set_optimization": True
        },
        "md": {
            "temperature": 300.0,
            "pressure": 0.0,
            "timestep": 0.001,
            "n_steps": 1000,
            "uncertainty_threshold": 5.0,
            "check_interval": 10
        },
        "workflow": {
            "max_iterations": 10,
            "convergence_energy": 0.001,
            "convergence_force": 0.01,
            "state_file_path": "state.json",
            "batch_size": 5,
            "n_candidates": 10,
            "checkpoint_interval": 1,
            "data_dir": "data",
            "active_learning_dir": "active_learning",
            "potentials_dir": "potentials"
        },
        "logging": {
            "level": "INFO",
            "log_file": "pyacemaker.log",
            "max_bytes": 10485760,
            "backup_count": 5
        }
    }

    # Simple deep merge for overrides
    for key, value in overrides.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
             base_config[key].update(value)  # type: ignore[attr-defined]
        else:
             base_config[key] = value

    return base_config
