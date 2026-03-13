import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError

from pyacemaker.domain_models import (
    DFTConfig,
    MDConfig,
    StructureConfig,
    TrainingConfig,
)
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.defaults import (
    DEFAULT_ACTIVE_LEARNING_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OTF_LOCAL_N_CANDIDATES,
    DEFAULT_OTF_LOCAL_N_SELECT,
    DEFAULT_OTF_MAX_RETRIES,
    DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
    DEFAULT_POTENTIALS_DIR,
    DEFAULT_STATE_FILE,
)
from pyacemaker.domain_models.structure import ExplorationPolicy
from tests.constants import TEST_ENERGY_GENERIC


@pytest.fixture(autouse=True)
def _mock_lammps_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the lammps module globally before any tests run.

    This is necessary because the LAMMPS python package (`lammps`) might attempt
    to load native C++ libraries or start an MPI environment upon import or
    instantiation, which can crash the test suite. By overriding the `lammps`
    key in `sys.modules`, any downstream file importing `lammps` will receive
    the mocked object instead.
    """
    monkeypatch.setitem(sys.modules, "lammps", MagicMock())


@pytest.fixture
def dummy_pseudopotentials_dir() -> Generator[Path, None, None]:
    """Provides a temporary directory with dummy pseudopotential files that cleans itself up."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def create_dummy_pseudopotentials(path: Path | str, elements: list[str]) -> None:
    """Helper to safely create dummy pseudopotential files for testing.

    Args:
        path: Directory path where the UPF files should be created.
              This should strictly be a temporary directory provided by a fixture.
        elements: List of chemical element symbols (e.g., ['Fe', 'Pt']).

    Raises:
        OSError: If there are permission issues or the directory doesn't exist.
    """
    try:
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        for el in elements:
            file_path = base_path / f"{el}.UPF"
            file_path.touch()
    except OSError as e:
        msg = f"Failed to create dummy pseudopotential files at {path}"
        raise OSError(msg) from e


@pytest.fixture
def mock_dft_config(dummy_pseudopotentials_dir: Path, monkeypatch: Any) -> DFTConfig:
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["H", "O", "Fe"])

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
    from pyacemaker.domain_models.md import ZBLConfig

    return MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        zbl=ZBLConfig(zbl_cut_inner=2.0, zbl_cut_outer=2.5),
    )


class MockCalculator(Calculator):
    """
    Mock ASE calculator for testing purposes.
    Can simulate failures and setup errors.

    Attributes:
        implemented_properties (list[str]): The properties this calculator can calculate.
        fail_count (int): The number of times to simulate calculation failure.
        setup_error (bool): Whether to throw an error on setup.
        attempts (int): Counter for the number of calculate calls.
        test_energy (float): The energy to return on successful calculation.
        results (dict[str, Any]): Internal storage for calculated values.
    """

    implemented_properties: list[str]
    fail_count: int
    setup_error: bool
    attempts: int
    test_energy: float
    results: dict[Any, Any]

    def __init__(
        self, fail_count: int = 0, setup_error: bool = False, test_energy: float | None = None
    ) -> None:
        """
        Initializes the mock calculator.

        Args:
            fail_count: How many iterations should simulate a calculation failure.
            setup_error: Whether the calculator should fail during setup (`calculate()`).
            test_energy: The energy value returned upon successful calculation.
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.implemented_properties = ["energy", "forces", "stress"]
        self.fail_count = fail_count
        self.setup_error = setup_error
        self.attempts = 0
        self.test_energy = test_energy if test_energy is not None else TEST_ENERGY_GENERIC
        self.results = {}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """
        Execute the mock calculation.

        Args:
            atoms: The Atoms object representing the structure to calculate.
            properties: The list of properties requested for this calculation.
            system_changes: List of strings detailing changes since last call.

        Raises:
            CalculatorSetupError: If `self.setup_error` is True.
            RuntimeError: If `self.attempts` is less than or equal to `self.fail_count`.
        """
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
    """Represents configuration specifically for structural generation and policies."""

    elements: list[str]
    supercell_size: list[int]
    policy_name: str


class DFTDict(TypedDict, total=False):
    """Configuration associated with the First Principles engine (Quantum Espresso)."""

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
    """Settings dictating how the MACE/ACE potentials are trained."""

    potential_type: str
    cutoff_radius: float
    max_basis_size: int
    delta_learning: bool
    active_set_optimization: bool


class MDDict(TypedDict, total=False):
    """Molecular Dynamics loop configuration parameters."""

    temperature: float
    pressure: float
    timestep: float
    n_steps: int
    uncertainty_threshold: float
    check_interval: int


class OTFDict(TypedDict, total=False):
    """On-The-Fly settings controlling uncertainty tolerances during MD."""

    uncertainty_threshold: float
    local_n_candidates: int
    local_n_select: int
    max_retries: int


class WorkflowDict(TypedDict, total=False):
    """Top-level execution and operational settings, logging directories."""

    max_iterations: int
    state_file_path: str
    active_learning_dir: str
    potentials_dir: str
    n_candidates: int
    batch_size: int
    otf: OTFDict


class ConfigDictType(TypedDict, total=False):
    """Complete root configuration dictionary representation for testing."""

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

    Args:
        **overrides: Key-value pairs matching PyAceConfig structure. Values
        can be dictionaries containing nested structures to override default
        values safely without disrupting validation.

    Returns:
        ConfigDictType: A structurally typed dictionary representation of
        the configuration, properly validated using Pydantic.

    Raises:
        ValueError: If validation fails when assembling the config.
    """
    if not isinstance(overrides, dict):
        msg = "Overrides must be dict"
        raise TypeError(msg)

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
            "pseudopotentials": {"Fe": "Fe.UPF"},
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
            "pressure": 1.0,
            "timestep": 0.001,
            "n_steps": 1000,
            "zbl": {"zbl_cut_inner": 1.0, "zbl_cut_outer": 1.5},
            "uncertainty_threshold": DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
            "check_interval": DEFAULT_CHECKPOINT_INTERVAL,
        },
        "validation": {},
        "workflow": {
            "max_iterations": 10,
            "state_file_path": DEFAULT_STATE_FILE,
            "active_learning_dir": DEFAULT_ACTIVE_LEARNING_DIR,
            "potentials_dir": DEFAULT_POTENTIALS_DIR,
            "n_candidates": DEFAULT_N_CANDIDATES,
            "batch_size": DEFAULT_BATCH_SIZE,
            "otf": {
                "uncertainty_threshold": DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
                "local_n_candidates": DEFAULT_OTF_LOCAL_N_CANDIDATES,
                "local_n_select": DEFAULT_OTF_LOCAL_N_SELECT,
                "max_retries": DEFAULT_OTF_MAX_RETRIES,
            },
        },
        "logging": {},
    }

    try:
        from pydantic import ValidationError

        # Merge dict properly utilizing Pydantic config
        model = PyAceConfig.model_validate(defaults)

        # Pydantic will perform standard merging and validation
        if overrides:
            model = model.model_copy(update=overrides, deep=True)

    except ValidationError as e:
        msg = f"Failed to merge test overrides due to validation constraints: {e}"
        raise ValueError(msg) from e
    else:
        return cast(ConfigDictType, model.model_dump())
