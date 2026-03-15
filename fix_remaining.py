import re

files_to_fix = [
    "tests/unit/test_eon_driver.py",
    "tests/unit/test_process.py",
    "tests/unit/mock_process.py",
    "tests/unit/test_report.py",
    "tests/unit/test_telemetry_pubsub.py",
    "tests/unit/test_path_validation.py"
]

for file in files_to_fix:
    try:
        with open(file, "r") as f:
            content = f.read()

        # fix common signatures
        content = re.sub(r'def (\w+)\(self, (.+)\):', r'def \1(self, \2) -> Any:', content)
        content = re.sub(r'def test_([a-zA-Z0-9_]+)\((.*?)\):', r'def test_\1(\2) -> None:', content)

        # specific manual replacements
        content = content.replace("def test_run_success(mock_popen):", "def test_run_success(mock_popen: Any) -> None:")
        content = content.replace("def test_run_failure(mock_popen):", "def test_run_failure(mock_popen: Any) -> None:")
        content = content.replace("def test_run_timeout(mock_popen):", "def test_run_timeout(mock_popen: Any) -> None:")

        # for EON driver mock
        content = content.replace("class MockProcessRunner(ProcessRunner):", "from pyacemaker.interfaces.process import ProcessRunner\nclass MockProcessRunner(ProcessRunner):")
        content = content.replace("def __init__(self, should_fail=False):", "def __init__(self, should_fail: bool = False) -> None:")
        content = content.replace("self.commands = []", "self.commands: list[list[str]] = []")

        with open(file, "w") as f:
            f.write(content)
    except FileNotFoundError:
        pass

