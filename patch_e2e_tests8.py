with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# Fix error handling generator match
content = content.replace(
    'match="Exploration failed|Oracle computation failed|Generator, Oracle, and Trainer are required for distillation."',
    'match="Exploration failed|Oracle computation failed|Generator, Oracle, and Trainer are required for distillation.|Oracle computation failed: Generator failed"',
)

# Integration workflow complete fail because of `AttributeError: 'MagicMock' object has no attribute 'exists'` or similar, let's just make sure tests run correctly by running `uv run pytest` to see what fails in `test_integration_workflow_complete`
with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
