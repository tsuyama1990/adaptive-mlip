from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError

from pyacemaker.domain_models import (
    DFTConfig,
    HybridParams,
    LoggingConfig,
    MDConfig,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    WorkflowConfig,
)
from pyacemaker.domain_models.structure import ExplorationPolicy
from tests.constants import TEST_ENERGY_GENERIC


def create_dummy_pseudopotentials(path: Any, elements: list[str]) -> None:
    """Helper to create dummy pseudopotential files."""
    for el in elements:
        (path / f"{el}.UPF").touch()


@pytest.fixture
def mock_dft_config(tmp_path: Any, monkeypatch: Any) -> DFTConfig:
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["H", "O", "Fe"])

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

@pytest.fixture
def mock_structure_config() -> StructureConfig:
    return StructureConfig(
        elements=["Fe"],
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.COLD_START,
    )

@pytest.fixture
def mock_training_config() -> TrainingConfig:
    return TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=500,
        delta_learning=True,
        active_set_optimization=False
    )

@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        hybrid_params=HybridParams(zbl_cut_inner=2.0, zbl_cut_outer=2.5)
    )


class MockCalculator(Calculator):
    """
    Mock ASE calculator for testing purposes.
    Can simulate failures and setup errors.
    """

    def __init__(self, fail_count: int = 0, setup_error: bool = False, test_energy: float | None = None) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.implemented_properties = ["energy", "forces", "stress"]
        self.fail_count = fail_count
        self.setup_error = setup_error
        self.attempts = 0
        self.test_energy = test_energy if test_energy is not None else TEST_ENERGY_GENERIC

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
            "energy": self.test_energy,
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
        policy_name=ExplorationPolicy.COLD_START,
        # adaptive_ratio, defect_density, strain_range removed
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
        active_set_optimization=False
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
