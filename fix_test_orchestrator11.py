import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    'assert mock_state_instance.mode == WORKFLOW_MODE_DISTILLATION',
    'assert mock_state_instance.state.mode == WORKFLOW_MODE_DISTILLATION'
)

with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)
