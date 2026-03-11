import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Add a mock incremental_train dynamically if we can't get the class to work.
content = content.replace("orch.trainer = FakeTrainer(refined_pot)", "orch.trainer = FakeTrainer(refined_pot)\n        orch.trainer.incremental_train = lambda *args, **kwargs: refined_pot")
content = content.replace("orch.trainer = FakeTrainer(tmp_path / \"pot\")", "orch.trainer = FakeTrainer(tmp_path / \"pot\")\n        orch.trainer.incremental_train = lambda *args, **kwargs: tmp_path / \"pot\"")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
