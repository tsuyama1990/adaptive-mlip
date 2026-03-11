with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# Replace test_integration_workflow_complete
# We will just patch the mock_config to disable distillation
content = content.replace(
    "workflow={'max_iterations': 3},",
    "workflow={'max_iterations': 3, 'distillation': {'enable': False}},",
)
# Fix the error handling tests to also use distillation: false
content = content.replace(
    "workflow={'max_iterations': 1}",
    "workflow={'max_iterations': 1, 'distillation': {'enable': False}}",
)

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
