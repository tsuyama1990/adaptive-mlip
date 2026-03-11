import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Since we replaced the class previously, the "assert new_pot == refined_pot" failed because of how MagicMock handles things vs the actual FakeTrainer class. Oh actually I used a MagicMock. Let's fix test_orchestrator_refinement_logic

content = content.replace("from unittest.mock import MagicMock\n\n    orch.trainer.incremental_train = MagicMock(return_value=refined_pot)", "")
content = content.replace("assert new_pot == refined_pot", "assert new_pot == refined_pot") # Actually `new_pot` was `None` because the refinement failed internally.

# Why did refinement fail?
# "ValueError: MACE model path /app/awakened_mace_model.model is outside allowed directory /app/potentials"
# Ah! FinetuneManager returns "awakened_mace_model.model", which is an invalid path for MACEManager because MACEManager strictly validates paths to be inside `potentials`.

# We need to mock FinetuneManager to return a valid path inside potentials_dir!

content = content.replace("patch(\"pyacemaker.core.oracle.MACEManager\") as MockMACEManager:\n", "patch(\"pyacemaker.core.oracle.MACEManager\") as MockMACEManager:\n        MockFinetuneManager.return_value.finetune.return_value = str(Path(config.workflow.potentials_dir) / \"awakened_mace_model.model\")\n        (Path(config.workflow.potentials_dir) / \"awakened_mace_model.model\").touch()\n")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
