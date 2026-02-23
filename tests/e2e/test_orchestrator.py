import pytest
from unittest.mock import Mock, patch
from typing import List, Iterator, Any, Union
from pathlib import Path
from ase import Atoms
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.domain_models import PyAceConfig, WorkflowConfig, StructureConfig, DFTConfig, TrainingConfig, MDConfig, LoggingConfig
from pyacemaker.core.base import BaseGenerator, BaseOracle, BaseTrainer, BaseEngine
from pyacemaker.constants import LOG_COMPUTED_PROPERTIES, LOG_ITERATION_COMPLETED, LOG_POTENTIAL_TRAINED

# Concrete Fakes for testing
class FakeGenerator(BaseGenerator):
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            yield Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])

class FakeOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info['energy'] = -10.0
            yield atoms

class FakeTrainer(BaseTrainer):
    def train(self, training_data_path: Union[str, Path]) -> Any:
        # Verify the file exists and is not empty
        path = Path(training_data_path)
        if not path.exists() or path.stat().st_size == 0:
            msg = "Training data file missing or empty"
            raise RuntimeError(msg)
        return "fake_potential.yace"

class FakeEngine(BaseEngine):
    def run(self, structure: Atoms, potential: Any) -> Any:
        return "simulation_result"

@pytest.fixture
def mock_config(tmp_path: Path) -> PyAceConfig:
    # Use tmp_path for logs and state file
    # Ensure log file path is valid and relative/resolvable to CWD.
    # In tests, we often change CWD or use tmp_path.
    # If LoggingConfig validation checks relative to CWD, we must ensure tmp_path is inside CWD OR mock the check?
    # Or simpler: in tests, we can just use "test.log" which is relative to whatever CWD is.
    return PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1,1,1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(
            max_iterations=2,
            state_file_path=str(tmp_path / "test_state.json"),
            n_candidates=10,
            batch_size=2
        ),
        # Using a simple relative path ensures it passes validation against CWD
        logging=LoggingConfig(level="DEBUG", log_file="test.log")
    )

def test_orchestrator_initialization(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)
    assert orch.iteration == 0
    # Orchestrator resolves state file path
    assert orch.state_file.name == "test_state.json"

def test_orchestrator_loop_with_fakes(mock_config: PyAceConfig, tmp_path: Path, caplog: Any) -> None:
    # Mocking os.getcwd only affects python level, but path traversal check uses path.resolve().
    # path.resolve() resolves symlinks and relative paths.
    # If tmp_path is in /tmp, and project is in /app, is_relative_to will fail if CWD is /app.
    # We must ensure we run the test logic inside tmp_path OR ensure config paths are inside /app.
    # Let's change CWD to tmp_path for this test.

    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    # We need to recreate config because LoggingConfig validation ran at fixture creation time relative to OLD cwd.
    # So we must create config INSIDE this test after chdir.
    config = PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1,1,1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(
            max_iterations=2,
            state_file_path="test_state.json", # Relative
            n_candidates=10,
            batch_size=2
        ),
        logging=LoggingConfig(level="DEBUG", log_file="test.log") # Relative
    )

    try:
        orch = Orchestrator(config)

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

        # Verify data flow
        data_dir = Path("data") # relative to CWD (tmp_path)
        assert data_dir.exists()
        training_file = data_dir / "training_iter_1.xyz"
        assert training_file.exists()
        assert training_file.stat().st_size > 0

        # Verify logging
        assert LOG_COMPUTED_PROPERTIES.format(count=10) in caplog.text
        assert LOG_POTENTIAL_TRAINED in caplog.text
        assert LOG_ITERATION_COMPLETED.format(iteration=1) in caplog.text

    finally:
        os.chdir(original_cwd)

def test_orchestrator_checkpointing(mock_config: PyAceConfig, tmp_path: Path) -> None:
    # 1. Save state
    # Use relative paths for Orchestrator logic
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Recreate config relative to new CWD
    config = PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1,1,1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(max_iterations=10, state_file_path="state.json"),
        logging=LoggingConfig(log_file="test.log")
    )

    try:
        orch1 = Orchestrator(config)
        orch1.iteration = 5
        orch1.save_state()

        # 2. Load state
        orch2 = Orchestrator(config)
        assert orch2.iteration == 5
    finally:
        os.chdir(original_cwd)

def test_orchestrator_error_handling(mock_config: PyAceConfig, tmp_path: Path) -> None:
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Recreate config
    config = PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1,1,1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100),
        workflow=WorkflowConfig(max_iterations=2),
        logging=LoggingConfig(log_file="test.log")
    )

    try:
        orch = Orchestrator(config)
        mock_gen = Mock(spec=BaseGenerator)
        mock_gen.generate.side_effect = RuntimeError("Generator failed")
        orch.generator = mock_gen

        with pytest.raises(RuntimeError):
            orch.run()
    finally:
        os.chdir(original_cwd)
