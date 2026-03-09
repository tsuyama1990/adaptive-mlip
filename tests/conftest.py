import sys
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError

from pyacemaker.domain_models import (
    DFTConfig,
    HybridParams,
    MDConfig,
    StructureConfig,
    TrainingConfig,
)
from pyacemaker.domain_models.structure import ExplorationPolicy
from tests.constants import TEST_ENERGY_GENERIC


@pytest.fixture(autouse=True)
def _mock_lammps_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lammps", MagicMock())


def create_dummy_pseudopotentials(path: Path | str, elements: list[str]) -> None:
    """Helper to create dummy pseudopotential files."""
    base_path = Path(path)
    for el in elements:
        file_path = base_path / f"{el}.UPF"
        file_path.touch()


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
        active_set_optimization=False,
    )


@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        hybrid_params=HybridParams(zbl_cut_inner=2.0, zbl_cut_outer=2.5),
    )


class MockCalculator(Calculator):
    """
    Mock ASE calculator for testing purposes.
    Can simulate failures and setup errors.
    """

    def __init__(
        self, fail_count: int = 0, setup_error: bool = False, test_energy: float | None = None
    ) -> None:
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


class StructureDict(TypedDict, total=False):
    elements: list[str]
    supercell_size: list[int]
    policy_name: str


class DFTDict(TypedDict, total=False):
    code: str
    functional: str
    kpoints_density: float
    encut: float
    pseudopotentials: dict[str, str]
    mixing_beta: float
    smearing_type: str
    smearing_width: float
    diagonalization: str


class TrainingDict(TypedDict, total=False):
    potential_type: str
    cutoff_radius: float
    max_basis_size: int
    delta_learning: bool
    active_set_optimization: bool


class MDDict(TypedDict, total=False):
    temperature: float
    pressure: float
    timestep: float
    n_steps: int
    uncertainty_threshold: float
    check_interval: int


class OTFDict(TypedDict, total=False):
    uncertainty_threshold: float
    local_n_candidates: int
    local_n_select: int
    max_retries: int


class WorkflowDict(TypedDict, total=False):
    max_iterations: int
    state_file_path: str
    active_learning_dir: str
    potentials_dir: str
    n_candidates: int
    batch_size: int
    otf: OTFDict


class ConfigDictType(TypedDict, total=False):
    project_name: str
    structure: StructureDict
    dft: DFTDict
    training: TrainingDict
    md: MDDict
    validation: dict[str, Any]
    workflow: WorkflowDict
    logging: dict[str, Any]


def create_test_config_dict(**overrides: Any) -> ConfigDictType:
    """
    Helper to create a valid config dictionary using Pydantic defaults.
    Constructs dictionary first to allow overrides before validation.
    """
    # 1. Create default dictionary structure (NOT Pydantic objects yet)
    # Use dummy value for validation that might fail if file doesn't exist,
    # but we rely on overrides to fix it or loose validation context where applicable.
    # Actually, we should use safe defaults for Pydantic construction if possible,
    # but validation is strict.

    defaults: dict[str, Any] = {
        "project_name": "TestProject",
        "structure": {
            "elements": ["Fe"],
            "supercell_size": [1, 1, 1],
            "policy_name": ExplorationPolicy.COLD_START,
        },
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 500.0,
            "pseudopotentials": {"Fe": "Fe.UPF"},  # Expects file to exist
            "mixing_beta": 0.7,
            "smearing_type": "mv",
            "smearing_width": 0.1,
            "diagonalization": "david",
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 5.0,
            "max_basis_size": 500,
            "delta_learning": True,
            "active_set_optimization": False,
        },
        "md": {
            "temperature": 300.0,
            "pressure": 0.0,
            "timestep": 0.001,
            "n_steps": 1000,
            "uncertainty_threshold": 5.0,
            "check_interval": 10,
        },
        "validation": {},  # Use defaults
        "workflow": {
            "max_iterations": 10,
            "state_file_path": "state.json",
            "active_learning_dir": "active_learning",
            "potentials_dir": "potentials",
            "n_candidates": 10,
            "batch_size": 5,
            "otf": {
                "uncertainty_threshold": 5.0,
                "local_n_candidates": 20,
                "local_n_select": 5,
                "max_retries": 3,
            },
        },
        "logging": {},
    }

    # 2. Apply overrides (Deep merge)
    for key, value in overrides.items():
        if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
            defaults[key].update(value)
        else:
            defaults[key] = value

    return defaults  # type: ignore[return-value]
