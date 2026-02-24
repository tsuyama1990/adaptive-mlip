from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.loop import LoopState, LoopStatus
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.md import MDSimulationResult
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def uat_config(tmp_path: Path) -> PyAceConfig:
    (tmp_path / "Fe.UPF").touch()
    config_dict = {
        "project_name": "UAT_Project",
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


def test_scenario_06_01_active_learning_campaign(uat_config: PyAceConfig, tmp_path: Path) -> None:
    """
    Scenario 06-01: Verify that the system can run a complete active learning loop from start to finish (mocked).
    """
    with patch("pyacemaker.orchestrator.setup_logger"), \
         patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_factory:

        # Mock modules
        mock_gen = MagicMock()
        mock_oracle = MagicMock()
        mock_trainer = MagicMock()
        mock_engine = MagicMock()
        mock_selector = MagicMock()
        mock_validator = MagicMock()

        mock_factory.return_value = (mock_gen, mock_oracle, mock_trainer, mock_engine, mock_selector, mock_validator)

        # Pre-create potential files
        pot1 = tmp_path / "pot_v1.yace"
        pot2 = tmp_path / "pot_v2.yace"
        pot3 = tmp_path / "pot_v3.yace"
        pot1.touch()
        pot2.touch()
        pot3.touch()

        # Setup behaviors
        mock_gen.generate.return_value = iter([Atoms("Fe")])
        mock_oracle.compute.return_value = iter([Atoms("Fe")])
        mock_trainer.train.side_effect = [pot1, pot2, pot3]

        # Iteration 1: Halt
        halt_path = tmp_path / "halt1.xyz"
        write(halt_path, Atoms("Fe"))

        res1 = MDSimulationResult(
            energy=-10.0, temperature=300, forces=[[0.0, 0.0, 0.0]], n_steps=50, max_gamma=10.0, halted=True,
            halt_structure_path=str(halt_path)
        )

        # Iteration 2: Converged (not halted)
        res2 = MDSimulationResult(
            energy=-10.0, temperature=300, forces=[[0.0, 0.0, 0.0]], n_steps=1000, max_gamma=2.0, halted=False,
            halt_structure_path=None
        )

        mock_engine.run.side_effect = [res1, res2]

        mock_gen.generate_local.return_value = iter([Atoms("Fe")])
        mock_selector.select.return_value = iter([Atoms("Fe")])

        # Run Orchestrator
        orch = Orchestrator(uat_config)
        orch.run()

        # Expectations
        # 1. Loop runs for exactly 2 iterations (max_iterations=2)
        assert orch.loop_state.iteration == 2
        # Check calls
        assert mock_engine.run.call_count == 2
        assert mock_trainer.train.call_count >= 2


def test_scenario_06_02_resume_capability(uat_config: PyAceConfig, tmp_path: Path) -> None:
    """
    Scenario 06-02: Verify that the system can resume a campaign after interruption.
    """
    state_file = Path(uat_config.workflow.state_file_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Pre-create state file: Iteration 1 finished.
    # We want to resume to start iteration 2.
    current_pot = tmp_path / "pot_v1.yace"
    current_pot.touch()

    state = LoopState(iteration=1, status=LoopStatus.RUNNING, current_potential=current_pot)
    state.save(state_file)

    with patch("pyacemaker.orchestrator.setup_logger"), \
         patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_factory:

        mock_gen = MagicMock()
        mock_oracle = MagicMock()
        mock_trainer = MagicMock()
        mock_engine = MagicMock()
        mock_selector = MagicMock()
        mock_validator = MagicMock()
        mock_factory.return_value = (mock_gen, mock_oracle, mock_trainer, mock_engine, mock_selector, mock_validator)

        # Iteration 2: Run MD
        res2 = MDSimulationResult(
            energy=-10.0, temperature=300, forces=[[0.0, 0.0, 0.0]], n_steps=1000, max_gamma=2.0, halted=False,
            halt_structure_path=None
        )
        mock_engine.run.return_value = res2

        # Run
        orch = Orchestrator(uat_config)
        orch.run()

        # Expectations
        assert orch.loop_state.iteration == 2
        mock_engine.run.assert_called_once() # Only 1 run
