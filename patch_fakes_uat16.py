import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Since FakeTrainer is finally instantiated, it returns the exact path now correctly:
# "return self.output_pot" which is `tmp_path / "refined.yace"`.

# Why is `new_pot` None? Oh, the `MockMACEManager` path issue.
# `ValueError: MACE model path /app/awakened_mace_model.model is outside allowed directory /app/potentials`
# So we need `MockFinetuneManager` to return a path in `potentials_dir`.
# `config.workflow.potentials_dir` is `str(tmp_path / "pots")`
# But we never patched `FinetuneManager.finetune` properly, because in `patch_fakes_uat15.py` the indentation was wrong, or something.

content = content.replace("MockFinetuneManager.return_value.finetune.return_value = str(Path(config.workflow.potentials_dir) / \"awakened_mace_model.model\")\n        (Path(config.workflow.potentials_dir) / \"awakened_mace_model.model\").touch()\n", "mace_path = Path(config.workflow.potentials_dir) / \"awakened_mace_model.model\"\n        mace_path.parent.mkdir(parents=True, exist_ok=True)\n        mace_path.touch()\n        MockFinetuneManager.return_value.finetune.return_value = str(mace_path.resolve())\n")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
