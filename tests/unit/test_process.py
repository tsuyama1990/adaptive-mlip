import shutil
import subprocess

import pytest

from pyacemaker.utils.process import run_command


def test_run_command_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "success", ""))
    monkeypatch.setattr(shutil, "which", lambda x: "/bin/ls")

    res = run_command(["ls", "-l"])
    assert res.returncode == 0
    assert res.stdout == "success"

def test_run_command_empty():
    with pytest.raises(ValueError, match="Command list cannot be empty"):
        run_command([])

def test_run_command_not_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda x: None)
    with pytest.raises(FileNotFoundError, match="Executable not found"):
        run_command(["nonexistent_command"])

def test_run_command_shell_injection(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda x: "/bin/echo")

    with pytest.raises(ValueError, match="potentially dangerous"):
        run_command(["echo", "hello", "&&", "rm", "-rf", "/"])

    with pytest.raises(ValueError, match="potentially dangerous"):
        run_command(["echo", "hello", ";", "ls"])

    with pytest.raises(ValueError, match="potentially dangerous"):
        run_command(["echo", "$(ls)"])

    with pytest.raises(ValueError, match="potentially dangerous"):
        run_command(["echo", "`ls`"])

    with pytest.raises(ValueError, match="potentially dangerous"):
        run_command(["echo", "hello", "|", "grep", "h"])
