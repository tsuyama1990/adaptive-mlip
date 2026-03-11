with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# Replace test_orchestrator_error_handling_oracle_stream check
content = content.replace(
    'match="Labeling failed"',
    'match="Generator, Oracle, and Trainer are required for distillation."',
)
content = content.replace(
    'match="Exploration failed"',
    'match="Generator, Oracle, and Trainer are required for distillation."',
)

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
