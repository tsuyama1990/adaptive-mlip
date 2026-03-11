import sys
import re

# `test_orchestrator_refinement_logic` in `tests/e2e/test_orchestrator_refinement.py` is failing. Let's see why.
with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Since we updated `_refine_potential` to compute DFT using `self.oracle.dft`, the mock oracle needs to have a `.dft` attribute.
content = content.replace("orchestrator.oracle = MagicMock()", "orchestrator.oracle = MagicMock()\n    orchestrator.oracle.dft = MagicMock()")
content = content.replace("assert mock_oracle.compute.call_count >= 1", "assert orchestrator.oracle.dft.compute.call_count >= 1")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
