import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.utils.process import run_command


def test_run_command_success():
    """Test successful command execution."""
    with patch("pyacemaker.utils.process.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = run_command(["echo", "hello"])

        mock_run.assert_called_once_with(
            ["echo", "hello"],
            cwd=None,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        assert result == mock_result


def test_run_command_dangerous_characters():
    """Test that command with dangerous characters raises ValueError."""
    dangerous_cmds = [
        ["echo", "hello;"],
        ["ls", "&"],
        ["cat", "file|"],
        ["echo", "`ls`"],
        ["echo", "$USER"],
    ]

    for cmd in dangerous_cmds:
        with pytest.raises(ValueError, match="Argument contains potentially dangerous characters"):
            run_command(cmd)


def test_run_command_long_arguments(caplog):
    """Test that long arguments are truncated in the log."""
    long_arg = "a" * 150
    cmd = ["echo", long_arg]

    with caplog.at_level(logging.DEBUG), patch("pyacemaker.utils.process.subprocess.run"):
        run_command(cmd)

    assert "[TRUNCATED]" in caplog.text
    assert long_arg[:20] in caplog.text
    assert len(caplog.text) < 150  # Make sure the full string wasn't logged


def test_run_command_called_process_error(caplog):
    """Test that CalledProcessError is caught, logged, and re-raised."""
    cmd = ["false"]
    error = subprocess.CalledProcessError(1, cmd, stderr="Command failed")

    with (
        patch("pyacemaker.utils.process.subprocess.run", side_effect=error),
        pytest.raises(subprocess.CalledProcessError),
    ):
        run_command(cmd)

    assert "Command failed: false" in caplog.text
    assert "Exit code: 1" in caplog.text
    assert "Stderr: Command failed" in caplog.text


def test_run_command_file_not_found_error(caplog):
    """Test that FileNotFoundError is caught, logged, and re-raised."""
    cmd = ["nonexistent_command"]
    error = FileNotFoundError("No such file or directory")

    with (
        patch("pyacemaker.utils.process.subprocess.run", side_effect=error),
        pytest.raises(FileNotFoundError),
    ):
        run_command(cmd)

    assert "Executable not found: nonexistent_command" in caplog.text
