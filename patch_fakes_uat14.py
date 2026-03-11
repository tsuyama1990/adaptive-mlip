import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Fix MagicMock import for the class variable
content = content.replace("class FakeTrainer:\n    config = MagicMock()", "class FakeTrainer:\n    pass")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
