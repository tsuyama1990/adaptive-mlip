import sys
import re

with open("tests/uat/test_cycle06_uat.py", "r") as f:
    content = f.read()

content = content.replace("patch(\"pyacemaker.orchestrator.MACEManager\"), \\", "patch(\"pyacemaker.core.oracle.MACEManager\"), \\")

with open("tests/uat/test_cycle06_uat.py", "w") as f:
    f.write(content)

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

content = content.replace("patch(\"pyacemaker.orchestrator.MACEManager\") as MockMACEManager", "patch(\"pyacemaker.core.oracle.MACEManager\") as MockMACEManager")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
