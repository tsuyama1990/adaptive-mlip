import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from ase import Atoms

from pyacemaker.constants import (
    LOG_COMPUTED_PROPERTIES,
    LOG_ITERATION_COMPLETED,
    LOG_POTENTIAL_TRAINED,
)
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
from pyacemaker.factory import ModuleFactory
from pyacemaker.orchestrator import Orchestrator


# Concrete Fakes for testing
class FakeGenerator(BaseGenerator):
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            yield Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


class FakeOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -10.0
            yield atoms


class FakeTrainer(BaseTrainer):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def train(self, training_data_path: str | Path) -> Any:
        # Verify the file exists and is not empty
        path = Path(training_data_path)
        if not path.exists() or path.stat().st_size == 0:
            msg = "Training data file missing or empty"
            raise RuntimeError(msg)

        # Create a dummy potential file so deployment works
        pot_path = self.output_dir / "fake_potential.yace"
        pot_path.touch()
        return pot_path


class FakeEngine(BaseEngine):
    def run(self, structure: Atoms | None, potential: Any) -> Any:
        return {"status": "success", "trajectory": "path/to/traj"}


@pytest.fixture
def mock_config(tmp_path: Path) -> PyAceConfig:
    return PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0),
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
        # Use simple name for log file, logging setup will resolve it relative to CWD (tmp_path in test)
        logging=LoggingConfig(level="DEBUG", log_file="test.log"),
    )


def test_orchestrator_initialization(mock_config: PyAceConfig) -> None:
    orch = Orchestrator(mock_config)
    assert orch.iteration == 0
    assert orch.state_file.name == "test_state.json"


def test_integration_workflow_complete(
    mock_config: PyAceConfig, tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """Comprehensive integration test for the full active learning loop."""

    # Safely change CWD to tmp_path to test relative path logic without polluting /app
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    # Recreate config relative to new CWD for logging validation
    # (Although mock_config uses absolute paths for workflow, logging is relative)
    config = mock_config.model_copy()

    try:
        # Mock factory to return our fakes
        def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any]:
            return (
                FakeGenerator(),
                FakeOracle(),
                FakeTrainer(output_dir=tmp_path),
                FakeEngine(),
            )

        monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

        orch = Orchestrator(config)
        orch.run()

        assert orch.iteration == 2
        assert orch.state_file.exists()
        assert "iteration" in orch.state_file.read_text()

        # Verify data flow (New Directory Structure)
        active_learning_dir = Path(config.workflow.active_learning_dir)
        assert active_learning_dir.exists()

        iter_dir = active_learning_dir / "iter_001"
        assert iter_dir.exists()
        assert (iter_dir / "candidates").exists()
        assert (iter_dir / "dft_calc").exists()
        assert (iter_dir / "training").exists()
        assert (iter_dir / "md_run").exists()

        training_file = iter_dir / "training" / "training_data.xyz"
        assert training_file.exists()
        # Check if file content looks like XYZ (ase write)
        content = training_file.read_text()
        assert "Lattice" in content or "Properties" in content

        # Verify deployed potential
        potentials_dir = Path(config.workflow.potentials_dir)
        assert potentials_dir.exists()
        assert (potentials_dir / "generation_001.yace").exists()

        # Verify logging
        assert LOG_COMPUTED_PROPERTIES.format(count=10) in caplog.text
        assert LOG_POTENTIAL_TRAINED in caplog.text
        assert LOG_ITERATION_COMPLETED.format(iteration=1) in caplog.text

    finally:
        os.chdir(original_cwd)


def test_orchestrator_checkpointing(mock_config: PyAceConfig) -> None:
    # 1. Save state
    orch1 = Orchestrator(mock_config)
    orch1.iteration = 5
    orch1.save_state()

    # 2. Load state
    orch2 = Orchestrator(mock_config)
    assert orch2.iteration == 5


def test_orchestrator_corrupted_state_file(mock_config: PyAceConfig, tmp_path: Path, caplog: Any) -> None:
    """Test resilience against corrupted state file."""
    state_file = Path(mock_config.workflow.state_file_path)
    state_file.write_text("{invalid_json")

    orch = Orchestrator(mock_config)
    # Should warn and default to iteration 0
    assert orch.iteration == 0
    assert "Failed to load state" in caplog.text or "state load fail" in caplog.text.lower()


def test_orchestrator_error_handling_generator(mock_config: PyAceConfig, monkeypatch: Any) -> None:
    mock_gen = Mock(spec=BaseGenerator)
    mock_gen.generate.side_effect = RuntimeError("Generator failed")

    def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any]:
        return mock_gen, None, None, None

    monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

    orch = Orchestrator(mock_config)
    with pytest.raises(RuntimeError):
        orch.run()


def test_orchestrator_error_handling_oracle_stream(
    mock_config: PyAceConfig, tmp_path: Path, monkeypatch: Any
) -> None:
    # Safely change CWD to tmp_path to test relative path logic without polluting /app
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        # Oracle that fails during iteration
        class FailingOracle(BaseOracle):
            def compute(
                self, structures: Iterator[Atoms], batch_size: int = 10
            ) -> Iterator[Atoms]:
                msg = "Oracle computation failed"
                raise RuntimeError(msg)

        def mock_create_modules(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any]:
             return FakeGenerator(), FailingOracle(), None, None

        monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules)

        orch = Orchestrator(mock_config)

        # Should crash cleanly
        with pytest.raises(RuntimeError, match="Oracle computation failed"):
            orch.run()
    finally:
        os.chdir(original_cwd)
