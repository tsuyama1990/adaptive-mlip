with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# Fix error handling generator match
content = content.replace(
    'match="Exploration failed|Oracle computation failed|Generator, Oracle, and Trainer are required for distillation.|Oracle computation failed: Generator failed"',
    'match="Exploration failed: Generator failed"',
)

# Fix integration workflow complete test:
# The distillation code now does `paths["training"] / FILENAME_TRAINING` and writes to it.
# The mock trainer needs to return a string path but maybe it returns a Mock object.
# I'll just check test_integration_workflow_complete to see what fails exactly using pytest -k

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
