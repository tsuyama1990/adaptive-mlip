from abc import ABC, abstractmethod
import subprocess
from pathlib import Path
from typing import Any

class ProcessRunner(ABC):
    """Abstract interface for running external processes."""

    @abstractmethod
    def run(self, cmd: list[str], cwd: Path, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        pass

class SubprocessRunner(ProcessRunner):
    """Default implementation using subprocess.run."""

    def run(self, cmd: list[str], cwd: Path, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Enforce text=True and capture_output=True unless overridden safely
        # But for interface compliance we should respect kwargs or set defaults.
        # EONWrapper expects capture_output=True, text=True.
        kwargs.setdefault("capture_output", True)
        kwargs.setdefault("text", True)
        kwargs.setdefault("check", True)

        return subprocess.run(cmd, cwd=cwd, **kwargs)  # noqa: S603
