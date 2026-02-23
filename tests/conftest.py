from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError

from pyacemaker.domain_models import (
    DFTConfig,
    LoggingConfig,
    MDConfig,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    WorkflowConfig,
)


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
    """
    Helper to create a valid config dictionary using Pydantic defaults.
    Ensures configuration is always valid and synced with domain models.
    """
    # 1. Create default components
    structure = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        adaptive_ratio=0.5,
        defect_density=0.01,
        strain_range=0.05
    )
    dft = DFTConfig(
        code="qe",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"Fe": "Fe.UPF"},
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david"
    )
    training = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=500,
        delta_learning=True,
        active_set_optimization=True
    )
    md = MDConfig(
        temperature=300.0,
        pressure=0.0,
        timestep=0.001,
        n_steps=1000,
        uncertainty_threshold=5.0,
        check_interval=10
    )
    workflow = WorkflowConfig(
        max_iterations=10,
        state_file_path="state.json",
        active_learning_dir="active_learning",
        potentials_dir="potentials"
    )
    logging = LoggingConfig()

    # 2. Assemble full config
    full_config = PyAceConfig(
        project_name="TestProject",
        structure=structure,
        dft=dft,
        training=training,
        md=md,
        workflow=workflow,
        logging=logging
    )

    # 3. Export to dict
    config_dict = full_config.model_dump()

    # 4. Apply overrides (Simple deep merge)
    for key, value in overrides.items():
        if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
             config_dict[key].update(value)
        else:
             config_dict[key] = value

    return config_dict
