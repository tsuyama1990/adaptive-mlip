from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from ase import Atoms

from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models import (
    DFTConfig,
    LoggingConfig,
    MDConfig,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    WorkflowConfig,
)
from pyacemaker.orchestrator import Orchestrator


# Concrete Fakes for testing
class FakeGenerator(BaseGenerator):
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            yield Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


class FakeOracle(BaseOracle):
    def compute(self, structures: list[Atoms], batch_size: int = 10) -> list[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -10.0
        return structures


class FakeTrainer(BaseTrainer):
    def train(self, training_data: list[Atoms]) -> Any:
        return "fake_potential.yace"


class FakeEngine(BaseEngine):
    def run(self, structure: Atoms, potential: Any) -> Any:
        return "simulation_result"


@pytest.fixture
def mock_config(tmp_path: Path) -> PyAceConfig:
    # Use tmp_path for logs and state file
    return PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(
            max_iterations=2,
            state_file_path=str(tmp_path / "test_state.json"),  # Configurable state file
        ),
        logging=LoggingConfig(level="DEBUG", log_file=str(tmp_path / "test.log")),
    )


def test_orchestrator_initialization(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)
    assert orch.iteration == 0
    # Verify state file path is taken from config
    assert orch.state_file.name == "test_state.json"


def test_orchestrator_loop_with_fakes(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)

    # Inject Fakes
    orch.generator = FakeGenerator()
    orch.oracle = FakeOracle()
    orch.trainer = FakeTrainer()
    orch.engine = FakeEngine()

    # Run loop
    orch.run()

    assert orch.iteration == 2
    assert orch.state_file.exists()
    assert "iteration" in orch.state_file.read_text()


def test_orchestrator_checkpointing(mock_config: PyAceConfig) -> None:
    # 1. Save state
    orch1 = Orchestrator(mock_config)
    orch1.iteration = 5
    orch1.save_state()

    # 2. Load state
    orch2 = Orchestrator(mock_config)
    # Orchestrator loads state in __init__
    assert orch2.iteration == 5


def test_orchestrator_error_handling(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)
    mock_gen = Mock(spec=BaseGenerator)
    mock_gen.generate.side_effect = RuntimeError("Generator failed")
    orch.generator = mock_gen

    with pytest.raises(RuntimeError):
        orch.run()
