from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from ase import Atoms

from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.exceptions import OrchestratorError
from pyacemaker.domain_models import (
    DFTConfig,
    LoggingConfig,
    MDConfig,
    MDSimulationResult,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    WorkflowConfig,
)
from pyacemaker.domain_models.defaults import (
    FILENAME_TRAINING,
    LOG_COMPUTED_PROPERTIES,
    LOG_ITERATION_COMPLETED,
    LOG_POTENTIAL_TRAINED,
)
from pyacemaker.factory import ModuleFactory
from pyacemaker.orchestrator import Orchestrator


# Concrete Fakes for testing
class FakeGenerator(BaseGenerator):
    def __init__(self, elements: list[str] | None = None) -> None:
        self.elements = elements or ["H"]

    def update_config(self, config: Any) -> None:
        pass

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            symbol = self.elements[0]
            yield Atoms(f"{symbol}2", positions=[[0, 0, 0], [0, 0, 0.74]])

    def generate_local(self, base_structure: Atoms, n_candidates: int, **kwargs: Any) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            yield base_structure.copy()  # type: ignore[no-untyped-call]


class FakeOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -10.0
            yield atoms


class FakeTrainer(BaseTrainer):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def train(
        self,
        training_data_path: str | Path,
        initial_potential: str | Path | None = None
    ) -> Any:
        path = Path(training_data_path)
        if not path.exists():
            msg = "Training data file missing"
            raise RuntimeError(msg)

        pot_path = self.output_dir / "fake_potential.yace"
        pot_path.touch()
        return pot_path


class FakeEngine(BaseEngine):
    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        return MDSimulationResult(
             energy=-10.0,
             forces=[[0.0, 0.0, 0.0]],
             halted=False,
             max_gamma=0.0,
             n_steps=100,
             temperature=300.0,
             trajectory_path="traj.xyz",
             log_path="log.lammps",
             halt_structure_path=None
        )

    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
        return self.run(structure, potential)

    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        return structure


@pytest.fixture
def mock_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PyAceConfig:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "H.UPF").touch()

    return PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1], policy_name="cold_start"),
        dft=DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=400.0,
            pseudopotentials={"H": "H.UPF"},
        ),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(
            max_iterations=2,
            state_file_path=str(tmp_path / "test_state.json"),
            n_candidates=10,
            batch_size=2,
            data_dir=str(tmp_path / "data"),
            active_learning_dir=str(tmp_path / "active_learning"),
            potentials_dir=str(tmp_path / "potentials"),
        ),
        logging=LoggingConfig(level="DEBUG", log_file="test.log"),
    )


def test_orchestrator_initialization(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)
    assert orch.loop_state.iteration == 0
    assert orch.state_manager.state_file.name == "test_state.json"


def test_integration_workflow_complete(
    mock_config: PyAceConfig, tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """Comprehensive integration test for the full active learning loop."""
    config = mock_config.model_copy()

    def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        return (
            FakeGenerator(elements=cfg.structure.elements),
            FakeOracle(),
            FakeTrainer(output_dir=tmp_path),
            FakeEngine(),
            MagicMock(),
            MagicMock(),
            None,
        )

    monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

    orch = Orchestrator(config)
    orch.run()

    assert orch.loop_state.iteration == 2
    assert orch.state_manager.state_file.exists()
    assert "iteration" in orch.state_manager.state_file.read_text()

    active_learning_dir = Path(config.workflow.active_learning_dir)
    assert active_learning_dir.exists()

    iter0_dir = active_learning_dir / "iter_000"
    assert iter0_dir.exists()
    training_file = iter0_dir / "training" / FILENAME_TRAINING
    assert training_file.exists()
    content = training_file.read_text()
    assert "Lattice" in content or "Properties" in content

    iter1_dir = active_learning_dir / "iter_001"
    assert iter1_dir.exists()

    potentials_dir = Path(config.workflow.potentials_dir)
    assert potentials_dir.exists()
    assert (potentials_dir / "generation_001.yace").exists()

    assert LOG_COMPUTED_PROPERTIES.format(count=10) in caplog.text
    assert LOG_POTENTIAL_TRAINED in caplog.text
    assert LOG_ITERATION_COMPLETED.format(iteration=1) in caplog.text


def test_orchestrator_checkpointing(mock_config: PyAceConfig) -> None:
    orch1 = Orchestrator(mock_config)
    orch1.loop_state.iteration = 5
    orch1.state_manager.save()

    orch2 = Orchestrator(mock_config)
    assert orch2.loop_state.iteration == 5


def test_orchestrator_corrupted_state_file(mock_config: PyAceConfig, tmp_path: Path, caplog: Any) -> None:
    state_file = Path(mock_config.workflow.state_file_path)
    state_file.write_text("{invalid_json")

    orch = Orchestrator(mock_config)
    assert orch.loop_state.iteration == 0


def test_orchestrator_directory_creation_error(mock_config: PyAceConfig, monkeypatch: Any) -> None:
    """Test error handling when directory creation fails."""
    # Patch pathlib.Path.mkdir to raise PermissionError
    original_mkdir = Path.mkdir
    def mock_mkdir(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        if "iter_" in str(self):
            msg = "Mock permission denied"
            raise PermissionError(msg)
        original_mkdir(self, mode, parents, exist_ok)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    # We also need to mock module creation to pass init
    def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        return (FakeGenerator(), FakeOracle(), FakeTrainer(Path()), FakeEngine(), MagicMock(), MagicMock(), None)
    monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

    orch = Orchestrator(mock_config)

    # run() -> _check_initial_potential() -> _setup_iteration_directory(0) -> _ensure_directory -> mkdir
    # The transaction should catch the PermissionError, log, and re-raise as OrchestratorError (wrapping) or propagate?
    # _setup_iteration_directory raises OrchestratorError from Exception
    with pytest.raises(OrchestratorError, match="Failed to setup directory"):
        orch.run()


def test_orchestrator_error_handling_generator(mock_config: PyAceConfig, monkeypatch: Any) -> None:
    mock_gen = Mock(spec=BaseGenerator)
    mock_gen.generate.side_effect = RuntimeError("Generator failed")

    def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        return mock_gen, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None

    monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

    orch = Orchestrator(mock_config)
    with pytest.raises(OrchestratorError, match="Exploration failed"):
        orch.run()


def test_orchestrator_error_handling_oracle_stream(
    mock_config: PyAceConfig, monkeypatch: Any
) -> None:
    class FailingOracle(BaseOracle):
        def compute(
            self, structures: Iterator[Atoms], batch_size: int = 10
        ) -> Iterator[Atoms]:
            msg = "Oracle computation failed"
            raise RuntimeError(msg)

    def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
         return FakeGenerator(elements=cfg.structure.elements), FailingOracle(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None

    monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

    orch = Orchestrator(mock_config)
    with pytest.raises(OrchestratorError, match="Labeling failed"):
        orch.run()
