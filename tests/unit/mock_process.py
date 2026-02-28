from pathlib import Path
from unittest.mock import MagicMock

from pyacemaker.interfaces.process import ProcessRunner


class MockProcessRunner(ProcessRunner):
    """Mock runner for testing."""

    def __init__(self, returncode=0, stdout="", stderr="") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.commands = []

    def run(self, cmd: list[str], cwd: Path, **kwargs):
        self.commands.append((cmd, cwd))
        mock_res = MagicMock()
        mock_res.returncode = self.returncode
        mock_res.stdout = self.stdout
        mock_res.stderr = self.stderr

        # subprocess.run raises if check=True and returncode != 0
        if kwargs.get("check", False) and self.returncode != 0:
            import subprocess

            raise subprocess.CalledProcessError(
                self.returncode, cmd, output=self.stdout, stderr=self.stderr
            )

        return mock_res
