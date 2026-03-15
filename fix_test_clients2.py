import re

files = [
    "tests/integration/test_telemetry_endpoints.py",
    "tests/uat/test_telemetry_streaming_uat.py"
]

for filepath in files:
    with open(filepath, "r") as f:
        content = f.read()

    # We need to replace `with client.websocket_connect` with `with TestClient(app) as test_client, test_client.websocket_connect`
    # First, rename `client` to something else or just do `with TestClient(app) as test_client:`

    # Actually, we can just replace the global `client = TestClient(app)` with `def get_client(): return TestClient(app)`
    # And replace `with client.websocket_connect(` with `with TestClient(app) as client:\n        with client.websocket_connect(`

    # 1. remove global client
    content = content.replace("client = TestClient(app)", "")

    # 2. wrap websocket_connect
    content = content.replace("with client.websocket_connect(", "with TestClient(app) as client:\n        with client.websocket_connect(")

    # But now indentation is wrong for everything inside `with client.websocket_connect`.
    # It's better to just do: `with TestClient(app).websocket_connect(...) as websocket:`
    # WAIT! `TestClient(app)` as a context manager triggers the lifespan.
    # So `with TestClient(app) as client, client.websocket_connect(...) as websocket:`

    content = re.sub(r'with client\.websocket_connect\((.*?)\) as websocket:', r'with TestClient(app) as client, client.websocket_connect(\1) as websocket:', content)

    with open(filepath, "w") as f:
        f.write(content)
