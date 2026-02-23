from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.loop import LoopState
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.md import MDSimulationResult
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def mock_config(tmp_path: Path) -> PyAceConfig:
    (tmp_path / "Fe.UPF").touch()
    config_dict = {
        "project_name": "TestProject",
        "structure": {
            "elements": ["Fe"],
            "supercell_size": [1, 1, 1],
            "policy_name": "cold_start",
        },
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 500.0,
            "pseudopotentials": {"Fe": str(tmp_path / "Fe.UPF")},
            "mixing_beta": 0.7,
            "smearing_type": "mv",
            "smearing_width": 0.1,
            "diagonalization": "david",
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 5.0,
            "max_basis_size": 500,
            "delta_learning": True,
            "active_set_optimization": False,
        },
        "md": {
            "temperature": 300.0,
            "pressure": 0.0,
            "timestep": 0.001,
            "n_steps": 1000,
            "uncertainty_threshold": 5.0,
            "check_interval": 10,
        },
        "workflow": {
            "max_iterations": 2,
            "state_file_path": str(tmp_path / "state.json"),
            "data_dir": str(tmp_path / "data"),
            "active_learning_dir": str(tmp_path / "active_learning"),
            "potentials_dir": str(tmp_path / "potentials"),
        },
        "logging": {},
    }
    return PyAceConfig(**config_dict)


@pytest.fixture
def orchestrator(mock_config: PyAceConfig) -> Orchestrator:
    with patch("pyacemaker.orchestrator.setup_logger"):
        orch = Orchestrator(mock_config)

        # Mock components
        orch.generator = MagicMock()
        orch.oracle = MagicMock()
        orch.trainer = MagicMock()
        orch.engine = MagicMock()
        orch.active_set_selector = MagicMock()
        return orch


def test_cold_start(orchestrator: Orchestrator, tmp_path: Path) -> None:
    # Inject loop_state if not present (simulate new implementation)
    if not hasattr(orchestrator, "loop_state"):
        orchestrator.loop_state = LoopState()

    # Setup mocks
    orchestrator.generator.generate.return_value = iter([Atoms("Fe")])  # type: ignore[union-attr]
    orchestrator.oracle.compute.return_value = iter([Atoms("Fe")])  # type: ignore[union-attr]
    initial_pot = tmp_path / "initial.yace"
    initial_pot.touch()
    orchestrator.trainer.train.return_value = initial_pot  # type: ignore[union-attr]

    # Execute
    orchestrator._check_initial_potential()

    # Verify
    assert orchestrator.loop_state.current_potential == initial_pot
    orchestrator.generator.generate.assert_called_once()  # type: ignore[union-attr]
    orchestrator.oracle.compute.assert_called_once()  # type: ignore[union-attr]
    orchestrator.trainer.train.assert_called_once()  # type: ignore[union-attr]


def test_resume_capability(mock_config: PyAceConfig, tmp_path: Path) -> None:
    # Setup saved state
    state_file = Path(mock_config.workflow.state_file_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    pot_path = tmp_path / "pot.yace"
    state_file.write_text(f'{{"iteration": 1, "status": "RUNNING", "current_potential": "{pot_path}"}}')

    # Re-initialize orchestrator to load state
    with patch("pyacemaker.orchestrator.setup_logger"):
        # This relies on Orchestrator.__init__ loading the state
        new_orch = Orchestrator(mock_config)

    # This assertion expects the code to be implemented
    if hasattr(new_orch, "loop_state"):
        assert new_orch.loop_state.iteration == 1
        assert new_orch.loop_state.current_potential == pot_path
    else:
        pytest.fail("Orchestrator does not have loop_state attribute")


def test_run_loop_iteration_halt(orchestrator: Orchestrator, tmp_path: Path) -> None:
    # Inject loop_state
    if not hasattr(orchestrator, "loop_state"):
        orchestrator.loop_state = LoopState()

    orchestrator.loop_state.current_potential = tmp_path / "current.yace"
    orchestrator.loop_state.current_potential.touch()

    # Mock MD halt
    halt_path = tmp_path / "halt.xyz"
    write(halt_path, Atoms("Fe")) # Write valid atoms for read()

    result = MDSimulationResult(
        energy=-100.0,
        temperature=300.0,
        forces=[[0.0, 0.0, 0.0]],
        n_steps=50,
        max_gamma=10.0,
        halted=True,
        halt_structure_path=str(halt_path)
    )
    orchestrator.engine.run.return_value = result  # type: ignore[union-attr]

    # Mock refinement
    orchestrator.generator.generate_local.return_value = iter([Atoms("Fe")])  # type: ignore[union-attr]
    orchestrator.active_set_selector.select.return_value = iter([Atoms("Fe")])  # type: ignore[union-attr]
    orchestrator.oracle.compute.return_value = iter([Atoms("Fe")])  # type: ignore[union-attr]
    refined_pot = tmp_path / "refined.yace"
    refined_pot.touch()
    orchestrator.trainer.train.return_value = refined_pot  # type: ignore[union-attr]

    # Execute
    orchestrator._run_loop_iteration()

    # Verify
    assert orchestrator.loop_state.iteration == 1
    assert orchestrator.loop_state.current_potential == refined_pot
    orchestrator.engine.run.assert_called()  # type: ignore[union-attr]
    orchestrator.generator.generate_local.assert_called()  # type: ignore[union-attr]
    orchestrator.trainer.train.assert_called()  # type: ignore[union-attr]
