with open("tests/integration/test_telemetry_endpoints.py", "r") as f:
    text = f.read()

# Replace global client with context manager inside tests?
# Actually, the easiest fix is just use AsyncClient from httpx, or properly use TestClient(app) inside a with block in a fixture.
