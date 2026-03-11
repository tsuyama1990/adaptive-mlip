import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Let's just fix the indentation manually
lines = content.split("\n")
new_lines = []
for line in lines:
    if "orch.trainer.incremental_train = lambda" in line:
        # Match previous line's indent
        new_lines.append("        " + line.strip())
    else:
        new_lines.append(line)

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write("\n".join(new_lines))
