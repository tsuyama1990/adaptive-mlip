from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.base import BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models import MDSimulationResult, PyAceConfig
from pyacemaker.orchestrator import Orchestrator


# Mock components that support streaming
class StreamingGenerator(BaseGenerator):
    def __init__(self, total_atoms: int) -> None:
        self.total_atoms = total_atoms

    def update_config(self, config: Any) -> None:
        pass

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        for i in range(n_candidates):
            yield Atoms("H", positions=[[0, 0, 0]], info={"id": i})

    def generate_local(self, base: Atoms, n_candidates: int) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            yield base.copy()  # type: ignore[no-untyped-call]


class StreamingOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -1.0
            yield atoms


class MockTrainer(BaseTrainer):
    def __init__(self, potential_path: Path) -> None:
        self.potential_path = potential_path

    def train(self, training_data_path: Path, initial_potential: Path | None = None) -> Path:
        # Create dummy potential file
        self.potential_path.touch()
        return self.potential_path


@pytest.fixture
def scaling_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PyAceConfig:
    # Need to change working directory so relative path H.UPF is found by DFTConfig validator
    monkeypatch.chdir(tmp_path)

    # Minimal config
    from pyacemaker.domain_models import (
        DFTConfig,
        LoggingConfig,
        MDConfig,
        StructureConfig,
        TrainingConfig,
        WorkflowConfig,
    )

    (tmp_path / "H.UPF").touch()

    return PyAceConfig(
        project_name="ScalingTest",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.1,
            encut=300,
            pseudopotentials={"H": "H.UPF"},
        ),
        training=TrainingConfig(potential_type="ace", cutoff_radius=3.0, max_basis_size=10),
        md=MDConfig(temperature=300, pressure=0, timestep=1, n_steps=10),
        workflow=WorkflowConfig(
            max_iterations=1,
            state_file_path=str(tmp_path / "state.json"),
            n_candidates=100,  # Simulate larger dataset
            batch_size=10,  # Small batch size to force chunking
            data_dir=str(tmp_path),
            active_learning_dir=str(tmp_path / "al"),
            potentials_dir=str(tmp_path / "pot"),
        ),
        logging=LoggingConfig(level="DEBUG", log_file="test.log"),
    )


def test_orchestrator_streaming_behavior(scaling_config: PyAceConfig, tmp_path: Path) -> None:
    """Verify that Orchestrator uses batching and doesn't crash with streaming mocks."""

    # Real write_extxyz
    from ase.io.extxyz import write_extxyz

    with patch("pyacemaker.orchestrator.write_extxyz", side_effect=write_extxyz) as mock_write:
        # Setup mocks
        gen = StreamingGenerator(total_atoms=100)
        oracle = StreamingOracle()
        potential_path = tmp_path / "pot.yace"
        trainer = MockTrainer(potential_path)

        # Create a mock engine that returns valid simulation results
        # To avoid the TypeError during _refine_potential logic
        mock_engine = MagicMock()
        mock_engine.run.return_value = MDSimulationResult(
            energy=-10.0,
            forces=[[0.0, 0.0, 0.0]],
            halted=False,
            max_gamma=0.0,
            n_steps=1000,
            temperature=300.0,
            trajectory_path="traj.xyz",
            log_path="log.lammps",
            halt_structure_path=None,
        )

        # Patch factory
        with patch(
            "pyacemaker.factory.ModuleFactory.create_modules",
            return_value=(gen, oracle, trainer, mock_engine, MagicMock(), MagicMock()),
        ):
            orch = Orchestrator(scaling_config)
            orch.run()

            # Verification
            # n_candidates=100, batch_size=10 -> Should have 10 batches for exploration
            # And 10 batches for labeling
            # Total expected calls: 20

            assert mock_write.call_count >= 20

            # Verify batch sizes
            for call in mock_write.call_args_list:
                args, _ = call
                batch = args[1]
                assert len(list(batch)) <= 10  # Should respect batch size
