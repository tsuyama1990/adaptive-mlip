from pyacemaker.domain_models.state import GlobalState, StepState, WorkflowStatus


def test_workflow_status_enum():
    assert WorkflowStatus.PENDING == "PENDING"
    assert WorkflowStatus.FAILED == "FAILED"

def test_global_state_defaults():
    state = GlobalState()
    assert state.status == WorkflowStatus.PENDING
    assert state.current_step == 0
    assert state.steps == {}

def test_step_state_update():
    state = GlobalState()
    state.steps["step1"] = StepState(status=WorkflowStatus.RUNNING, message="Generating")
    assert state.steps["step1"].status == WorkflowStatus.RUNNING
