from unittest.mock import MagicMock, patch

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.workflow import WorkflowStep
from pyacemaker.orchestrator import Orchestrator


def test_integration_full_distillation_flow(
    mock_dft_config, mock_structure_config, mock_training_config, mock_md_config, tmp_path
) -> None:
    """
    End-to-end integration test of the 7-step workflow with mocked modules.
    Verifies the orchestration logic, state transitions, and error handling structure
    without running actual heavy computations.
    """
    # Setup config
    state_file = tmp_path / "state.json"
    al_dir = tmp_path / "al"
    pot_dir = tmp_path / "pot"
    data_dir = tmp_path / "data"

    config = PyAceConfig(
        project_name="integ_test",
        structure=mock_structure_config,
        dft=mock_dft_config,
        training=mock_training_config,
        md=mock_md_config,
        workflow={
            "max_iterations": 1,
            "state_file_path": str(state_file),
            "active_learning_dir": str(al_dir),
            "potentials_dir": str(pot_dir),
            "data_dir": str(data_dir),
            "n_candidates": 5
        },
        distillation={
            "enable_mace_distillation": True,
            "step1_direct_sampling": {"target_points": 10, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
            "step2_active_learning": {"uncertainty_threshold": 0.8, "dft_calculator": "VASP"},
            "step3_mace_finetune": {"base_model": "test"}
            }
    )

    # We patch create_modules to return mocks that behave "successfully"
    with patch("pyacemaker.factory.ModuleFactory.create_modules") as mock_create:
        mock_gen = MagicMock()
        # Return iterables as expected by _stream_write
        mock_gen.generate.return_value = iter([])
        mock_gen.generate_local.return_value = iter([])

        mock_oracle = MagicMock()
        mock_oracle.compute.return_value = iter([])

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = "dummy.yace"

        mock_selector = MagicMock()
        mock_selector.select.return_value = iter([])

        mock_create.return_value = (
            mock_gen, mock_oracle, mock_trainer, MagicMock(), mock_selector, MagicMock()
        )

        orchestrator = Orchestrator(config)
        orchestrator.run()

        # Verify state reached final step
        # Since _run_distillation_workflow iterates sequentially and saves state before execution:
        # 1. Sets Step 1, runs.
        # ...
        # 7. Sets Step 7 (DELTA_LEARNING), runs.
        # Loop finishes.

        # We need to reload state to verify persistence
        from pyacemaker.core.loop import LoopState
        loaded_state = LoopState.load(state_file)

        assert loaded_state.current_step == WorkflowStep.DELTA_LEARNING
        assert loaded_state.mode == "distillation"
