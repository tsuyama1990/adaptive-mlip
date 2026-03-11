import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Let's completely remove the injected lambda and just fix the class definition indentation.
content = re.sub(r'^\s+orch.trainer.incremental_train = lambda.*?\n', '', content, flags=re.MULTILINE)

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
