import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Fix indentation. The first replace had 8 spaces (because it was inside the function), the second should also have 8 spaces.
content = content.replace("orch.trainer.incremental_train = lambda *args, **kwargs: refined_pot", "orch.trainer.incremental_train = lambda *args, **kwargs: refined_pot") # That doesn't fix it if it added too much.
