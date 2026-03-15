import re

files = [
    "tests/integration/test_telemetry_endpoints.py",
    "tests/uat/test_telemetry_streaming_uat.py"
]

for filepath in files:
    with open(filepath, "r") as f:
        content = f.read()

    # We need to completely remove _reset_broker fixture
    content = re.sub(r'@pytest\.fixture\(autouse=True\)\ndef _reset_broker\(\) -> None:.*?(?=\ndef )', '', content, flags=re.DOTALL)

    with open(filepath, "w") as f:
        f.write(content)
