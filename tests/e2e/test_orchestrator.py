from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

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
            yield Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])

class FakeOracle(BaseOracle):
    def compute(self, structures: list[Atoms], batch_size: int = 10) -> list[Atoms]:
        # Return structures with dummy energy
        for atoms in structures:
            atoms.info['energy'] = -10.0
        return structures

class FakeTrainer(BaseTrainer):
    def train(self, training_data: list[Atoms]) -> Any:
        return "fake_potential.yace"

class FakeEngine(BaseEngine):
    def run(self, structure: Atoms, potential: Any) -> Any:
        return "simulation_result"

@pytest.fixture
def mock_config() -> PyAceConfig:
    return PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1,1,1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(max_iterations=2),
        logging=LoggingConfig(level="DEBUG")
    )

def test_orchestrator_initialization(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)
    assert orch.iteration == 0
    assert orch.state_file.name == "TestProject_state.json"

def test_orchestrator_loop_with_fakes(mock_config: PyAceConfig, tmp_path: Any) -> None:
    # Use tmp_path for state file to avoid clutter
    with patch("pyacemaker.orchestrator.Path", side_effect=lambda x: tmp_path / x):
        orch = Orchestrator(mock_config)

        # Inject Fakes
        orch.generator = FakeGenerator()
        orch.oracle = FakeOracle()
        orch.trainer = FakeTrainer()
        orch.engine = FakeEngine()

        # Run loop
        orch.run()

        assert orch.iteration == 2

        # Verify state file existence
        state_file = tmp_path / "TestProject_state.json"
        assert state_file.exists()
        assert "iteration" in state_file.read_text()

def test_orchestrator_checkpointing(mock_config: PyAceConfig, tmp_path: Any) -> None:
     with patch("pyacemaker.orchestrator.Path", side_effect=lambda x: tmp_path / x):
        orch = Orchestrator(mock_config)
        orch.iteration = 5
        orch.save_state()

        # New instance should load state
        orch2 = Orchestrator(mock_config)
        assert orch2.iteration == 5

def test_orchestrator_error_handling(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)

    # Correctly mock generator to raise exception on method call
    mock_gen = Mock(spec=BaseGenerator)
    mock_gen.generate.side_effect = RuntimeError("Generator failed")
    orch.generator = mock_gen

    with pytest.raises(RuntimeError):
        orch.run()
