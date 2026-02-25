from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.core.container import Container
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


# Fake Components
class FakeGenerator(BaseGenerator):
    def update_config(self, config: Any) -> None: pass
    def generate(self, n_candidates: int) -> Iterator[Atoms]: yield from []

    def generate_local(self, base_structure: Atoms, n_candidates: int, **kwargs: Any) -> Iterator[Atoms]:
        for _ in range(n_candidates):
            yield base_structure.copy()  # type: ignore[no-untyped-call]

class FakeOracle(BaseOracle):
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        for atoms in structures:
            atoms.info["energy"] = -5.0
            yield atoms

class FakeTrainer(BaseTrainer):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def train(self, training_data_path: str | Path, initial_potential: str | Path | None = None) -> Any:
        self.output_path.touch()
        return self.output_path

class FakeActiveSetSelector(ActiveSetSelector):
    def select(self, candidates: Any, potential_path: Any, n_select: int, anchor: Any = None) -> Iterator[Atoms]:
        if anchor:
            yield anchor
            n_select -= 1

        cands = list(candidates)
        for i in range(min(n_select, len(cands))):
            yield cands[i]

def test_orchestrator_refinement_logic(tmp_path: Path) -> None:
    # Create dummy UPF
    (tmp_path / "H.UPF").write_text("dummy UPF content")

    # 1. Setup Config
    config = PyAceConfig(
        project_name="TestRefine",
        structure=StructureConfig(elements=["H"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(
            code="qe",
            functional="PBE",
            pseudopotentials={"H": str(tmp_path / "H.UPF")},
            kpoints_density=0.04,
            encut=400.0
        ),
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

    # 2. Setup Orchestrator
    orch = Orchestrator(config)
    # Initialize container manually
    orch.container = MagicMock(spec=Container)

    # 4. Inject Fakes
    refined_pot = tmp_path / "refined.yace"
    orch.container.generator = FakeGenerator()
    orch.container.active_set_selector = FakeActiveSetSelector()
    orch.container.oracle = FakeOracle()
    orch.container.trainer = FakeTrainer(refined_pot)
    orch.container.engine = MagicMock()

    # 3. Create dummy halt structure with gamma
    halt_path = tmp_path / "halt.extxyz"
    atoms = Atoms("H2", positions=[[0,0,0], [5.0,0,0]], cell=[10,10,10], pbc=True)
    # Atom 1 (at 5.0) has high gamma
    atoms.new_array("c_gamma", np.array([0.1, 10.0])) # type: ignore[no-untyped-call]
    write(halt_path, atoms, format="extxyz")

    # 5. Create Simulation Result
    result = MDSimulationResult(
        energy=-10.0, forces=[[0,0,0]], stress=[0]*6, halted=True, max_gamma=10.0,
        n_steps=500, temperature=300,
        halt_structure_path=str(halt_path), halt_step=500
    )

    # 6. Run _refine_potential
    paths = {"training": tmp_path / "training"}
    paths["training"].mkdir(parents=True)

    new_pot = orch._refine_potential(result, Path("old.yace"), paths)

    assert new_pot == refined_pot

    assert any(paths["training"].iterdir())

def test_orchestrator_refinement_extraction_failure(tmp_path: Path, caplog: Any) -> None:
    # Test graceful handling of extraction failure

    (tmp_path / "H.UPF").write_text("dummy UPF content")

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
    orch.container = MagicMock(spec=Container)

    # Inject modules
    orch.container.generator = FakeGenerator()
    orch.container.active_set_selector = FakeActiveSetSelector()
    orch.container.oracle = FakeOracle()
    orch.container.trainer = FakeTrainer(tmp_path / "pot")
    orch.container.engine = MagicMock()

    halt_path = tmp_path / "halt.extxyz"
    atoms = Atoms("H", positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    atoms.new_array("c_gamma", np.array([10.0])) # type: ignore[no-untyped-call]
    write(halt_path, atoms, format="extxyz")

    result = MDSimulationResult(
        energy=0, forces=[[0.0, 0.0, 0.0]], stress=[0]*6, halted=True, max_gamma=10.0, n_steps=10, temperature=300,
        halt_structure_path=str(halt_path)
    )

    with pytest.MonkeyPatch.context() as m:
        def mock_fail(*args: Any, **kwargs: Any) -> None:
            msg = "Boom"
            raise ValueError(msg)

        m.setattr("pyacemaker.orchestrator.extract_local_region", mock_fail)

        new_pot = orch._refine_potential(result, Path("p"), {})

        assert new_pot is None
        assert "Failed to extract local cluster" in caplog.text
