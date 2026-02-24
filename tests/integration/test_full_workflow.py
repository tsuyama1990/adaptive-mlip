from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models import (
    DFTConfig,
    LoggingConfig,
    MDConfig,
    MDSimulationResult,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    ValidationConfig,
    WorkflowConfig,
)
from pyacemaker.factory import ModuleFactory
from pyacemaker.orchestrator import Orchestrator


# Concrete Fakes for integration testing
class FakeGenerator(BaseGenerator):
    def __init__(self, elements: list[str] | None = None) -> None:
        self.elements = elements or ["H"]

    def update_config(self, config: Any) -> None:
        pass

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        for i in range(n_candidates):
            symbol = self.elements[0]
            atoms = Atoms(f"{symbol}2", positions=[[0, 0, 0], [0, 0, 0.74]])
            atoms.info["source"] = f"gen_{i}"
            yield atoms

    def generate_local(self, base_structure: Atoms, n_candidates: int) -> Iterator[Atoms]:
        for i in range(n_candidates):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            atoms.info["source"] = f"local_{i}"
            yield atoms


class FakeOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms]) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -13.6
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

        pot_path = self.output_dir / "potential.yace"
        pot_path.touch()
        return pot_path


class FakeEngine(BaseEngine):
    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        # Simulate a stable run that completes
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


@pytest.fixture
def integration_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PyAceConfig:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "H.UPF").touch()

    return PyAceConfig(
        project_name="IntegrationTest",
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
            max_iterations=1,  # Only 1 iteration for speed
            state_file_path=str(tmp_path / "integration_state.json"),
            n_candidates=5,
            batch_size=2,
            data_dir=str(tmp_path / "data"),
            active_learning_dir=str(tmp_path / "active_learning"),
            potentials_dir=str(tmp_path / "potentials"),
        ),
        validation=ValidationConfig(), # Enable validation
        logging=LoggingConfig(level="DEBUG", log_file="integration.log"),
    )


def test_full_workflow_with_validation(
    integration_config: PyAceConfig, tmp_path: Path, monkeypatch: Any
) -> None:
    """
    Tests the complete workflow including the validation phase.
    Mocks the Validator to avoid running actual LAMMPS/Phonopy which are slow/complex to mock entirely.
    """
    config = integration_config.model_copy()

    # Mock Validator to return success
    mock_validator = MagicMock()
    mock_validator_instance = MagicMock()
    mock_validator_instance.validate.return_value = MagicMock(
        phonon_stable=True,
        elastic_stable=True,
        imaginary_frequencies=[],
        elastic_tensor=[[1.0]],
        bulk_modulus=100.0,
        shear_modulus=50.0,
        plots={}
    )
    mock_validator.return_value = mock_validator_instance

    def mock_create_modules_fixed(cfg: PyAceConfig) -> tuple[Any, Any, Any, Any, Any]:
        return (
            FakeGenerator(elements=cfg.structure.elements),
            FakeOracle(),
            FakeTrainer(output_dir=tmp_path / "potentials"),
            FakeEngine(),
            mock_validator # Return the class mock
        )

    monkeypatch.setattr(ModuleFactory, "create_modules", mock_create_modules_fixed)

    # Ensure dirs exist
    (tmp_path / "potentials").mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(config)
    orch.run()

    # Verify flow
    assert orch.loop_state.iteration == 1
    # Check that validation was attempted
    mock_validator.assert_called() # Check that Validator was instantiated
    mock_validator_instance.validate.assert_called() # Check that validate() was called

    # Check artifacts
    assert (tmp_path / "potentials" / "potential.yace").exists()
