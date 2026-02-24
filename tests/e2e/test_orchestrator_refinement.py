from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
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

    # 3. Create dummy halt structure with gamma
    halt_path = tmp_path / "halt.extxyz"
    # Create 2 atoms far apart to ensure clear center distinction
    atoms = Atoms("H2", positions=[[0,0,0], [5.0,0,0]], cell=[10,10,10], pbc=True)
    # Atom 1 (at 5.0) has high gamma
    atoms.new_array("c_gamma", np.array([0.1, 10.0]))  # type: ignore[no-untyped-call]
    write(halt_path, atoms, format="extxyz")

    # 4. Mock Modules
    orch.generator = MagicMock(spec=BaseGenerator)
    # generate_local returns iterator of atoms
    orch.generator.generate_local.return_value = iter([Atoms("H") for _ in range(3)])

    orch.active_set_selector = MagicMock(spec=ActiveSetSelector)
    orch.active_set_selector.select.return_value = iter([Atoms("H")])

    orch.oracle = MagicMock(spec=BaseOracle)
    orch.oracle.compute.return_value = iter([Atoms("H")])

    orch.trainer = MagicMock(spec=BaseTrainer)
    orch.trainer.train.return_value = tmp_path / "refined.yace"
    (tmp_path / "refined.yace").touch()

    # 5. Create Simulation Result
    result = MDSimulationResult(
        energy=-10.0, forces=[[0,0,0]], halted=True, max_gamma=10.0,
        n_steps=500, temperature=300,
        halt_structure_path=str(halt_path), halt_step=500
    )

    # 6. Run _refine_potential
    paths = {"training": tmp_path / "training"}
    paths["training"].mkdir(parents=True)

    new_pot = orch._refine_potential(result, Path("old.yace"), paths)

    assert new_pot == tmp_path / "refined.yace"

    # 7. Verify calls
    # Check if extract_local_region logic worked implicitly by checking args passed to active_set_selector
    _, kwargs = orch.active_set_selector.select.call_args
    assert "anchor" in kwargs
    anchor = kwargs["anchor"]
    assert isinstance(anchor, Atoms)

    # Verify the anchor corresponds to atom 1
    # Original atom 1 was at [5.0, 0, 0].
    # Extracted cluster centers atom 1.
    # In the cluster, position should be centered (embed_cluster behavior).
    # But weight should be 1.0.
    weights = anchor.get_array("force_weight")  # type: ignore[no-untyped-call]
    # If extraction radius is default (6.0), atom 0 (dist 5.0) is also included.
    # Atom 1 is center (weight 1.0). Atom 0 is dist 5.0 <= 6.0, so weight 1.0.
    # Note: Since cell is 10.0 and radius+buffer is 10.0, we see multiple images.
    # Atom 0 at 0.0 is dist 5.0 (left). Atom 0 at 10.0 is dist 5.0 (right).
    # Both are included. So 1 center + 2 neighbors = 3 atoms.
    assert len(weights) == 3

    # Verify generator called with anchor
    gen_args, _ = orch.generator.generate_local.call_args
    assert gen_args[0] == anchor
