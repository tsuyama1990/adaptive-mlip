import sys

files = [
    "tests/integration/test_telemetry_endpoints.py",
    "tests/uat/test_telemetry_streaming_uat.py"
]

for filepath in files:
    with open(filepath, "r") as f:
        content = f.read()

    # Instead of rewriting all logic, what if we just start the background task manually in the fixture?
    # Yes! In `_reset_broker()` we can manually trigger the background task!

    if "asyncio.create_task" not in content:
        replacement = """
@pytest.fixture(autouse=True)
def _reset_broker() -> None:
    telemetry_broker.queue = asyncio.Queue(maxsize=100)
    loop = asyncio.get_event_loop()
    telemetry_broker.initialize_loop(loop)
    from pyacemaker.api.routes.telemetry import broadcast_loop
    task = loop.create_task(broadcast_loop())
    yield
    task.cancel()
"""
        # Find the _reset_broker fixture and replace it
        import re
        content = re.sub(r'@pytest\.fixture\(autouse=True\)\ndef _reset_broker\(\) -> None:.*?(?=\ndef )', replacement, content, flags=re.DOTALL)

        with open(filepath, "w") as f:
            f.write(content)
