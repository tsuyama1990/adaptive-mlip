with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# Fix error handling generator
content = content.replace('match="Oracle computation failed"', 'match="Exploration failed"')

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
