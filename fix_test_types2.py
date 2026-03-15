from pathlib import Path

content = Path("tests/integration/test_api_endpoints.py").read_text()
# Ensure payload definitions are just typed as Any dicts
content = content.replace("base_payload =", "base_payload: dict[str, Any] =")
content = content.replace(
    "from pyacemaker.main import app", "from typing import Any\nfrom pyacemaker.main import app"
)
content = content.replace(
    "payload_speed = dict(base_payload)  # type: ignore", "payload_speed = dict(base_payload)"
)
content = content.replace(
    "payload_acc = dict(base_payload)  # type: ignore", "payload_acc = dict(base_payload)"
)
Path("tests/integration/test_api_endpoints.py").write_text(content)
