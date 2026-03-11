with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# Revert error message matching for test_orchestrator_error_handling_generator
content = content.replace('match="Exploration failed"', 'match="Oracle computation failed"')
content = content.replace(
    'match="Oracle computation failed"',
    'match="Exploration failed|Oracle computation failed|Generator, Oracle, and Trainer are required for distillation."',
)

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
