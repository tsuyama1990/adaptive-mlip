from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseOracle, BaseTrainer
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.dft import DFTConfig
from pyacemaker.domain_models.md import MDConfig, MDSimulationResult
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.domain_models.workflow import OTFConfig, WorkflowConfig
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def mock_config(tmp_path: Path) -> PyAceConfig:
    # Create dummy pseudopotential
    pseudo_path = tmp_path / "Fe.pbe.UPF"
    pseudo_path.write_text("<UPF version=\"2.0.1\">\nPP_HEADER\n")

    dft_config = DFTConfig(
        code="qe",
        functional="PBE",
        kpoints_density=0.03,
        encut=500,
        pseudopotentials={"Fe": str(pseudo_path)}
    )
    training_config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="test_pot.yace",
        delta_learning=True,
        elements=["Fe"],
        seed=123,
        max_iterations=100,
        batch_size=10
    )

    return PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["Fe"], supercell_size=[2, 2, 2]),
        dft=dft_config,
        training=training_config,
        md=MDConfig(temperature=300, pressure=0, timestep=0.001, n_steps=1000, fix_halt=True),
        workflow=WorkflowConfig(
            max_iterations=1,
            data_dir=str(tmp_path / "data"),
            active_learning_dir=str(tmp_path / "al"),
            potentials_dir=str(tmp_path / "pots"),
            otf=OTFConfig(max_retries=1, local_n_candidates=5, local_n_select=2)
        )
    )

@pytest.fixture
def mock_modules() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    generator = MagicMock(spec=BaseGenerator)
    oracle = MagicMock(spec=BaseOracle)
    trainer = MagicMock(spec=BaseTrainer)
    engine = MagicMock(spec=BaseEngine)
    active_set = MagicMock(spec=ActiveSetSelector)
    return generator, oracle, trainer, engine, active_set

def test_orchestrator_otf_halt_loop(
    mock_config: PyAceConfig,
    mock_modules: tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock],
    tmp_path: Path
) -> None:
    generator, oracle, trainer, engine, active_set = mock_modules

    # Mock return values
    halt_structure = Atoms("Fe", positions=[[0,0,0]], cell=[2,2,2], pbc=True)
    halt_path = tmp_path / "halt.xyz"
    write(halt_path, halt_structure)

    # Engine run sequence:
    # 1. Halt
    # 2. Success (after retraining)
    res_halt = MDSimulationResult(
        energy=-100.0, forces=[[0,0,0]], halted=True, max_gamma=10.0, n_steps=500, temperature=300,
        halt_structure_path=str(halt_path), log_path="log"
    )
    res_success = MDSimulationResult(
        energy=-100.0, forces=[[0,0,0]], halted=False, max_gamma=1.0, n_steps=1000, temperature=300,
        halt_structure_path=None, log_path="log"
    )
    engine.run.side_effect = [res_halt, res_success]

    # Generator
    local_candidates = [Atoms("Fe", cell=[2,2,2], pbc=True), Atoms("Fe", cell=[2,2,2], pbc=True)]
    generator.generate_local.return_value = iter(local_candidates)

    # Active Set
    selected = [Atoms("Fe", cell=[2,2,2], pbc=True)]
    active_set.select.return_value = iter(selected)

    # Oracle
    labelled = [Atoms("Fe", cell=[2,2,2], pbc=True)]
    oracle.compute.return_value = iter(labelled)

    # Trainer
    new_pot_path = tmp_path / "new_pot.yace"
    new_pot_path.touch()
    trainer.train.return_value = new_pot_path

    # Initial potential
    initial_pot = tmp_path / "init.yace"
    initial_pot.touch()

    # Create dummy candidate for initial structure
    candidates_file = Path(mock_config.workflow.active_learning_dir) / "iter_001" / "candidates" / "candidates.xyz"
    candidates_file.parent.mkdir(parents=True)
    write(candidates_file, Atoms("Fe", cell=[2,2,2], pbc=True))

    with patch("pyacemaker.orchestrator.ModuleFactory.create_modules", return_value=(generator, oracle, trainer, engine)):
        orchestrator = Orchestrator(mock_config)
        # Manually set active set selector since we patch create_modules but active_set is separate
        # But initialize_modules overwrites it. So we mock ActiveSetSelector class.

        with patch("pyacemaker.orchestrator.ActiveSetSelector", return_value=active_set):
             orchestrator.initialize_modules()

             # Call _run_otf_loop directly to test specific logic
             paths = orchestrator._setup_iteration_directory(1)
             orchestrator._run_otf_loop(paths, initial_pot)

    # Verifications
    assert engine.run.call_count == 2

    # Check Generator called with halt structure
    generator.generate_local.assert_called_once()
    args, _ = generator.generate_local.call_args
    # We can check simple properties if objects are copied
    assert len(args[0]) == len(halt_structure)

    # Check Active Set called
    active_set.select.assert_called_once()

    # Check Trainer called with fine tuning
    trainer.train.assert_called_once()
    _, kwargs = trainer.train.call_args
    assert kwargs.get("initial_potential") == initial_pot
