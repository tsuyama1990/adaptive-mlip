import re

files = [
    "tests/integration/test_telemetry_endpoints.py",
    "tests/uat/test_telemetry_streaming_uat.py"
]

for filepath in files:
    with open(filepath, "r") as f:
        content = f.read()

    # Re-read global client
    content = content.replace("client = TestClient(app)", "")

    # We can just change `with client.websocket_connect( f"/api/v1/...` to:
    # `with TestClient(app) as client, client.websocket_connect( f"/api/v1/...`
    # Python 3.9+ allows multiple context managers separated by commas.

    content = re.sub(r'with client\.websocket_connect\((.*?)\) as websocket:', r'with TestClient(app) as client, client.websocket_connect(\1) as websocket:', content)

    with open(filepath, "w") as f:
        f.write(content)
