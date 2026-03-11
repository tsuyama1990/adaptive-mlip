import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Fix mock for the finetune manager since that is also called now in `_refine_potential`.
content = content.replace("with patch(\"pyacemaker.orchestrator.extract_intelligent_cluster\") as mock_extract:", "with patch(\"pyacemaker.orchestrator.extract_intelligent_cluster\") as mock_extract, \\\n             patch(\"pyacemaker.orchestrator.FinetuneManager\") as MockFinetuneManager, \\\n             patch(\"pyacemaker.orchestrator.MACEManager\") as MockMACEManager:")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)

with open("tests/uat/test_cycle06_uat.py", "r") as f:
    content = f.read()

content = content.replace("patch(\"pyacemaker.factory.ModuleFactory.create_modules\") as mock_factory,", "patch(\"pyacemaker.factory.ModuleFactory.create_modules\") as mock_factory, \\\n            patch(\"pyacemaker.orchestrator.FinetuneManager\"), \\\n            patch(\"pyacemaker.orchestrator.MACEManager\"), \\\n            patch(\"pyacemaker.orchestrator.extract_intelligent_cluster\", return_value=Atoms(\"Fe\")),")

with open("tests/uat/test_cycle06_uat.py", "w") as f:
    f.write(content)
