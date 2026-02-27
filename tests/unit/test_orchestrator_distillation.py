import pytest
from unittest.mock import MagicMock, patch

from pyacemaker.core.loop import LoopStatus
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.workflow import WorkflowStep
from pyacemaker.domain_models.defaults import (
    WORKFLOW_MODE_DISTILLATION,
    LOG_STEP_1,
    LOG_STEP_7,
)


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock()
    config.project_name = "test_project"

    # Enable distillation
    config.distillation.enable_mace_distillation = True

    # Workflow config (accessed in __init__)
    config.workflow.max_iterations = 1
    config.workflow.state_file_path = "state.json"
    config.workflow.active_learning_dir = "active_learning"
    config.workflow.data_dir = "data"
    config.workflow.potentials_dir = "potentials"
    config.workflow.batch_size = 5
    config.workflow.checkpoint_interval = 1
    config.distillation.step1_direct_sampling.target_points = 100

    # Logging config
    config.logging = MagicMock()

    return config


@patch("pyacemaker.orchestrator.shutil")
@patch("pyacemaker.orchestrator.StateManager")
@patch("pyacemaker.orchestrator.DirectoryManager")
@patch("pyacemaker.orchestrator.setup_logger")
@patch("pyacemaker.orchestrator.ModuleFactory.create_modules")
def test_distillation_workflow_execution(
    mock_create_modules,
    mock_setup_logger,
    mock_dir_manager,
    mock_state_manager_cls,
    mock_shutil,
    mock_config
) -> None:
    # Setup mocks
    mock_state_instance = MagicMock()
    mock_state_instance.iteration = 0
    mock_state_instance.current_step = None
    mock_state_manager_cls.return_value = mock_state_instance

    mock_logger_instance = MagicMock()
    mock_setup_logger.return_value = mock_logger_instance

    # Mock module factory return
    mock_gen = MagicMock()
    mock_oracle = MagicMock()
    mock_trainer = MagicMock()
    mock_engine = MagicMock()
    mock_selector = MagicMock()
    mock_validator = MagicMock()

    mock_create_modules.return_value = (
        mock_gen,
        mock_oracle,
        mock_trainer,
        mock_engine,
        mock_selector,
        mock_validator,
    )

    orchestrator = Orchestrator(mock_config)

    # Run the workflow
    orchestrator.run()

    # Verify mode set to distillation
    assert mock_state_instance.mode == WORKFLOW_MODE_DISTILLATION

    # Verify that steps were executed
    mock_logger_instance.info.assert_any_call(LOG_STEP_1)
    mock_logger_instance.info.assert_any_call(LOG_STEP_7)

    # Verify logic: Step 1 calls generator
    mock_gen.generate.assert_called()

    assert mock_state_instance.save.call_count >= 7

    # Verify the final state
    assert mock_state_instance.current_step == WorkflowStep.DELTA_LEARNING


@patch("pyacemaker.orchestrator.shutil")
@patch("pyacemaker.orchestrator.StateManager")
@patch("pyacemaker.orchestrator.DirectoryManager")
@patch("pyacemaker.orchestrator.setup_logger")
@patch("pyacemaker.orchestrator.ModuleFactory.create_modules")
def test_distillation_workflow_resume(
    mock_create_modules,
    mock_setup_logger,
    mock_dir_manager,
    mock_state_manager_cls,
    mock_shutil,
    mock_config
) -> None:
    # Setup state to resume from Step 4
    mock_state_instance = MagicMock()
    mock_state_instance.iteration = 0
    mock_state_instance.current_step = WorkflowStep.SURROGATE_SAMPLING # Step 4
    mock_state_instance.mode = WORKFLOW_MODE_DISTILLATION
    mock_state_manager_cls.return_value = mock_state_instance

    mock_logger_instance = MagicMock()
    mock_setup_logger.return_value = mock_logger_instance

    mock_create_modules.return_value = (
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    orchestrator = Orchestrator(mock_config)
    orchestrator.run()

    # Verify Log Step 1 was NOT called
    log_calls = [call.args[0] for call in mock_logger_instance.info.call_args_list]
    assert LOG_STEP_1 not in log_calls
    assert LOG_STEP_7 in log_calls


@patch("pyacemaker.orchestrator.shutil")
@patch("pyacemaker.orchestrator.StateManager")
@patch("pyacemaker.orchestrator.DirectoryManager")
@patch("pyacemaker.orchestrator.setup_logger")
@patch("pyacemaker.orchestrator.ModuleFactory.create_modules")
def test_distillation_workflow_error_handling(
    mock_create_modules,
    mock_setup_logger,
    mock_dir_manager,
    mock_state_manager_cls,
    mock_shutil,
    mock_config
) -> None:
    # Setup state
    mock_state_instance = MagicMock()
    mock_state_instance.iteration = 0
    mock_state_instance.current_step = None
    # Ensure nested mock exists for status assignment
    mock_state_instance.state = MagicMock()
    mock_state_manager_cls.return_value = mock_state_instance

    mock_create_modules.return_value = (
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    orchestrator = Orchestrator(mock_config)

    # Mock Step 1 to raise exception
    orchestrator._step1_direct_sampling = MagicMock(side_effect=RuntimeError("Step failed"))

    # Run
    with pytest.raises(RuntimeError, match="Step failed"):
        orchestrator.run()

    # Verify status updated
    # MagicMock records property set
    assert mock_state_instance.state.status == LoopStatus.HALTED
    # Verify save called
    assert mock_state_instance.save.call_count >= 1
