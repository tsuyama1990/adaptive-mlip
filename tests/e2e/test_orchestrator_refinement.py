from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from ase.io import write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
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
             stress=[0.0] * 6,
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

class FakeActiveSetSelector(ActiveSetSelector):
    pass

def test_orchestrator_refinement_logic(tmp_path: Path, caplog: Any) -> None:
    # Test that refinement loop is triggered and halted structure is extracted

    # Create dummy UPF
    (tmp_path / "H.UPF").write_text("dummy UPF content")

    # Setup minimal config & orch
    config = PyAceConfig(
        project_name="TestRefine",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(code="qe", functional="PBE", pseudopotentials={"H": str(tmp_path / "H.UPF")}, kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100, fix_halt=True),
        workflow=WorkflowConfig(
            max_iterations=1,
            state_file_path=str(tmp_path / "state.json"),
            data_dir=str(tmp_path / "data"),
            active_learning_dir=str(tmp_path / "al"),
            potentials_dir=str(tmp_path / "pots"),
        ),
        logging=LoggingConfig(level="DEBUG"),
    )
    orch = Orchestrator(config)

    # Inject modules
    orch.generator = FakeGenerator()
    orch.active_set_selector = FakeActiveSetSelector()
    orch.oracle = FakeOracle()
    orch.trainer = FakeTrainer(tmp_path / "pots")

    # Mock Engine to return HALTED result
    orch.engine = MagicMock(spec=BaseEngine)
    halt_path = tmp_path / "halt.xyz"
    write(halt_path, Atoms("H", positions=[[0,0,0]], cell=[10,10,10], pbc=True))

    result = MDSimulationResult(
        energy=0, forces=[[]], stress=[0.0]*6, halted=True, max_gamma=10.0, n_steps=10, temperature=300,
        trajectory_path=str(halt_path),
        log_path=str(tmp_path / "log.lammps"),
        halt_structure_path=str(halt_path)
    )
    orch.engine.run.return_value = result

    # Mock extraction util
    with patch("pyacemaker.orchestrator.extract_local_region") as mock_extract:
        mock_extract.return_value = Atoms("H")

        # Mock other methods
        orch.active_set_selector.select = MagicMock(return_value=iter([Atoms("H")]))
        orch.oracle.compute = MagicMock(return_value=iter([Atoms("H")]))

        # Set initial state
        (tmp_path / "pots").mkdir(exist_ok=True)
        pot = tmp_path / "pots" / "initial.yace"
        pot.touch()
        orch.loop_state.current_potential = pot

        # Run ONE iteration logic
        orch._run_loop_iteration()

        # Verify refinement flow
        mock_extract.assert_called_once()
        # Verify local generation was called? Or just extraction?
        # The logic is: Halt -> Extract -> (Maybe generate local) -> Label -> Train
        # Actually _refine_potential_from_halt calls active_set_selector.select on extracted structures
        orch.active_set_selector.select.assert_called()
        orch.oracle.compute.assert_called()
        # orch.trainer.train called twice? Once for initial (mocked out in this test setup)
        # We manually set current_potential so only refine train call expected
        # Wait, _run_loop_iteration does exploration first if not refining?
        # If halted, it refines.

        assert "MD Halted at step 10. Triggering refinement." in caplog.text


def test_orchestrator_refinement_extraction_failure(tmp_path: Path, caplog: Any) -> None:
    # Test graceful handling of extraction failure

    # Create dummy UPF
    (tmp_path / "H.UPF").write_text("dummy UPF content")

    # Setup minimal config & orch
    config = PyAceConfig(
        project_name="TestRefine",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(code="qe", functional="PBE", pseudopotentials={"H": str(tmp_path / "H.UPF")}, kpoints_density=0.04, encut=400.0),
        training=TrainingConfig(potential_type="ace", cutoff_radius=4.0, max_basis_size=100),
        md=MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=100, fix_halt=True),
        workflow=WorkflowConfig(
            max_iterations=1,
            state_file_path=str(tmp_path / "state.json"),
            data_dir=str(tmp_path / "data"),
            active_learning_dir=str(tmp_path / "al"),
            potentials_dir=str(tmp_path / "pots"),
        ),
        logging=LoggingConfig(level="DEBUG"),
    )
    orch = Orchestrator(config)

    # Inject modules
    orch.generator = FakeGenerator()
    orch.active_set_selector = FakeActiveSetSelector()
    orch.oracle = FakeOracle()
    orch.trainer = FakeTrainer(tmp_path / "pot")

    # Create halt structure but force extraction failure by patching extraction util
    halt_path = tmp_path / "halt.extxyz"
    atoms = Atoms("H", positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    atoms.new_array("c_gamma", np.array([10.0])) # type: ignore[no-untyped-call]
    write(halt_path, atoms, format="extxyz")

    result = MDSimulationResult(
        energy=0, forces=[[]], stress=[0.0]*6, halted=True, max_gamma=10.0, n_steps=10, temperature=300,
        trajectory_path=str(halt_path),
        log_path=str(tmp_path / "log.lammps"),
        halt_structure_path=str(halt_path)
    )
    orch.engine = MagicMock()
    orch.engine.run.return_value = result

    # Patch extract to fail
    with patch("pyacemaker.orchestrator.extract_local_region", side_effect=ValueError("Bad file")):
        # Set initial potential
        (tmp_path / "pots").mkdir(exist_ok=True)
        pot = tmp_path / "pots" / "initial.yace"
        pot.touch()
        orch.loop_state.current_potential = pot

        # Run
        orch._run_loop_iteration()

        # Should catch error and log warning, then proceed to normal exploration?
        # Or just fail gracefully?
        # The logic:
        assert "Failed to extract local cluster" in caplog.text
