from pathlib import Path

content = Path("tests/integration/test_api_endpoints.py").read_text()
# Fix type errors in test payload assignments
content = content.replace(
    "payload_speed = dict(base_payload)", "payload_speed = dict(base_payload)  # type: ignore"
)
content = content.replace(
    "payload_acc = dict(base_payload)", "payload_acc = dict(base_payload)  # type: ignore"
)
Path("tests/integration/test_api_endpoints.py").write_text(content)
