from unittest.mock import Mock

from pyacemaker.domain_models import PyAceConfig, WorkflowConfig
from pyacemaker.orchestrator import Orchestrator


def test_orchestrator_initialization_placeholder() -> None:
    """
    Placeholder test until Orchestrator is implemented.
    This test will be updated to import Orchestrator and verify initialization.
    """


def test_orchestrator_runs() -> None:
    config = Mock(spec=PyAceConfig)
    config.project_name = "TestProject"
    config.workflow = Mock(spec=WorkflowConfig)
    config.workflow.max_iterations = 1

    orch = Orchestrator(config)
    orch.run()
    assert orch.iteration == 1
