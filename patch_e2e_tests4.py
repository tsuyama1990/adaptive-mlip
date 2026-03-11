with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# E2E test fails because they do not implement distillation but distillation is True by default.
# The error match is for `Labeling failed` because when distillation is True, it tries to compute and catch all exceptions inside `self.oracle.compute()` inside `_check_initial_potential()`?
# Actually distillation is NOT catching the error. The error is escaping `_check_initial_potential()` but since it does not catch `Exception` around it.
# Let's just fix the mock config fixture directly in tests/e2e/test_orchestrator.py

content = content.replace(
    "workflow={'max_iterations': 3, 'distillation': {'enable': False}},",
    "workflow=WorkflowConfig(max_iterations=3, distillation=DistillationConfig(enable=False)),",
)
content = content.replace(
    "workflow={'max_iterations': 1, 'distillation': {'enable': False}}",
    "workflow=WorkflowConfig(max_iterations=1, distillation=DistillationConfig(enable=False))",
)

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
